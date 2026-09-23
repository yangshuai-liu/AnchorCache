from __future__ import annotations

import torch

from .attention import group_attn
from .kv import KVState, LayeredKV
from .topology import GroupPlan

# The framework owns only the grouping and K/V handling of attention. Projection, qk norm, RoPE
# and output projection stay in the backend: they are model-specific and carry LoRA /
# quantization, which copying weights would silently drop.
#
#   full     run both main + frozen groups without touching reusable state
#   extract  run both groups and write post-RoPE K/V from VIS segments into reusable state
#   cached   run only the main group and read VIS K/V from reusable state; FROZEN segments
#            are not projected at all, which is where cached steps save compute
#
# The plan covers the frozen-group cases without branching:
#   hid > 0          q=kv=[vis, hid]   static text anchors
#   hid == 0         q=kv=[vis]        isolated cache
#   vis == hid == 0  no frozen group; the main group is full attention

MODES = ("full", "extract", "cached")


def _stack(names: list[str], src: dict[str, torch.Tensor]) -> torch.Tensor:
    missing = [n for n in names if n not in src]
    assert not missing, f"regions {missing} are missing tensors; required {names}, got {sorted(src)}"
    if len(names) == 1:
        return src[names[0]]
    return torch.cat([src[n] for n in names], dim=1)


def _split(out: torch.Tensor, names: list[str], ref: dict[str, torch.Tensor]) -> dict:
    res, off = {}, 0
    for n in names:
        ln = ref[n].shape[1]
        res[n] = out[:, off : off + ln]
        off += ln
    assert off == out.shape[1], f"split-back length mismatch: {off} vs {out.shape[1]}"
    return res


def _repeat_kv(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """GQA: [B, S, H_kv, D] -> [B, S, H, D].

    On the dense path, q and kv are concatenated into the same kernel invocation, so
    their head counts must match. The varlen/flash path supports GQA natively and
    should not go through this function.
    """
    h_kv = x.shape[2]
    if h_kv == n_heads:
        return x
    assert n_heads % h_kv == 0, f"heads {n_heads} is not an integer multiple of kv_heads {h_kv}"
    return x.repeat_interleave(n_heads // h_kv, dim=2)


def store_vis(
    kv: LayeredKV, layer: int, plan: GroupPlan, k: dict, v: dict, clone: bool = True
) -> None:
    """Concatenate VIS-segment K/V in plan.vis order and write it to reusable state.

    clone is not a defensive copy: k[name] is usually a view into a larger tensor
    (Qwen's img_k contains target+ref, and the sliced cond_k is only a view). Without
    cloning, memory for the entire hidden tensor remains live until the request ends.
    """
    if not plan.vis:
        return
    key = _stack(plan.vis, k)
    val = _stack(plan.vis, v)
    kv.put(layer, KVState(key.clone() if clone else key, val.clone() if clone else val))
    kv.meta.setdefault("vis_spans", [(n, k[n].shape[1]) for n in plan.vis])
    kv.num_vis_tokens = key.shape[1]


def _vis_offsets(kv: LayeredKV) -> dict[str, tuple[int, int]]:
    spans = kv.meta.get("vis_spans")
    assert spans, "reusable state is missing vis_spans; store_vis was not called during extract"
    out, off = {}, 0
    for name, ln in spans:
        out[name] = (off, off + ln)
        off += ln
    return out


def _main_kv_cached(
    plan: GroupPlan, k: dict, v: dict, kv: LayeredKV, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble in plan.main_kv order, using cache for VIS and freshly computed values for LIVE.

    The order must match the extract step: attention output is invariant to KV ordering,
    but the flash reduction order introduces ~1e-5 drift.
    """
    cached = kv.get(layer)
    off = _vis_offsets(kv)
    ks, vs = [], []
    for name in plan.main_kv:
        if plan.kv_is_vis(name):
            a, b = off[name]
            ks.append(cached.key[:, a:b])
            vs.append(cached.value[:, a:b])
        else:
            assert name in k, f"cached step is missing K for live region {name!r}"
            ks.append(k[name])
            vs.append(v[name])
    return torch.cat(ks, dim=1), torch.cat(vs, dim=1)


def live_regions(plan: GroupPlan) -> list[str]:
    """Regions that actually require projection during a cached step.

    Hidden states for VIS/HID are not passed in at all. The backend uses this to decide which projection / MLP / Norm operations to skip
    on this step.
    """
    seen, out = set(), []
    for n in list(plan.main_q) + [n for n in plan.main_kv if not plan.kv_is_vis(n)]:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def cached_attn(
    q: dict[str, torch.Tensor],
    k: dict[str, torch.Tensor],
    v: dict[str, torch.Tensor],
    plan: GroupPlan,
    *,
    mode: str = "full",
    kv: LayeredKV | None = None,
    layer: int = 0,
    mask: torch.Tensor | None = None,
    frozen_mask: torch.Tensor | None = None,
    backend: str | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Execute grouped attention according to plan.

    q/k/v: region name -> **post-RoPE** tensor of shape [B, S_r, H, D]. post-RoPE is
    a strict requirement: storing pre-RoPE would require the cached step to know the
    absolute positions of VIS segments, leaking layout into runtime (see the KVState
    explanation in kv.py).

    Returns outputs for every region covered by main_q + frozen_q, still shaped
    [B, S_r, H, D]; the backend performs its own output projection.
    """
    assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
    if mode != "full":
        assert kv is not None, f"mode={mode!r} requires reusable state"

    # GQA heads are repeated only for the kernel; reusable state keeps the H_kv heads.
    n_heads = q[plan.main_q[0]].shape[2]

    if mode == "cached":
        assert not plan.has_frozen_group, (
            "a cached step must not have a frozen group; generate the plan with group_plan(layout, cached=True)"
        )
        main_k, main_v = _main_kv_cached(plan, k, v, kv, layer)
    else:
        main_k, main_v = _stack(plan.main_kv, k), _stack(plan.main_kv, v)

    main_q = _stack(plan.main_q, q)
    main_k, main_v = _repeat_kv(main_k, n_heads), _repeat_kv(main_v, n_heads)
    out = _split(
        group_attn(main_q, main_k, main_v, mask=mask, backend=backend, scale=scale),
        plan.main_q,
        q,
    )

    if plan.has_frozen_group:
        fq = _stack(plan.frozen_q, q)
        fout = group_attn(
            fq,
            _repeat_kv(_stack(plan.frozen_kv, k), n_heads),
            _repeat_kv(_stack(plan.frozen_kv, v), n_heads),
            mask=frozen_mask,
            backend=backend,
            scale=scale,
        )
        out.update(_split(fout, plan.frozen_q, q))

    if mode == "extract":
        store_vis(kv, layer, plan, k, v)
    return out
