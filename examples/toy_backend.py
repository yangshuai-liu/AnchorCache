"""Demonstrate how much code is required to add a fourth backend.

This is AnchorCache's only meaningful success criterion for its extensibility goal: not a low
line count, but whether a new backend can be integrated without modifying Core.

This file is executable and requires no model weights:

    python examples/toy_backend.py

It implements a hypothetical backend with the sequence
[prompt(vis) | anchor(hid) | live | target(live)], continuous-vector predictions, and latent
states. The entire adapter is under 60 lines, all of which express model-specific semantics
rather than framework boilerplate.
"""

import torch
import torch.nn as nn

from anchorcache.core import ModelOutput, PreparedCondition
from anchorcache.runtime import CacheManager, RollingResidency
from anchorcache.sampling import CfgGuidance, rollout
from anchorcache.training import Distiller, StudentRollout, TeacherForced, mse
from anchorcache.transformer import (
    HID,
    LIVE,
    VIS,
    KVState,
    LayeredKV,
    Region,
    SeqLayout,
    dense_mask_of,
    group_attn,
    group_plan,
)

DIM, HEADS, LAYERS = 32, 4, 3


class ToyBackend(nn.Module):
    """A hypothetical model with its own semantics: three layers, each with attention and a residual connection."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.qkv = nn.ModuleList(nn.Linear(DIM, 3 * DIM, bias=False) for _ in range(LAYERS))
        self.out = nn.ModuleList(nn.Linear(DIM, DIM, bias=False) for _ in range(LAYERS))
        self.head = nn.Linear(DIM, DIM, bias=False)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)

    def split(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv_cur(x).chunk(3, dim=-1)
        return tuple(t.view(b, s, HEADS, DIM // HEADS) for t in (q, k, v))

    def qkv_cur(self, x):
        return self._cur(x)


# ── The adapter begins here and represents all work required to integrate a new backend ─────


class ToyAdapter:
    def __init__(self, net: ToyBackend, anchor: bool = True):
        self.net = net
        self.anchor = anchor

    # 1. This model's token layout: the only topology information that must be declared
    def layout(self, prompt: int, live: int, target: int) -> SeqLayout:
        regions = [Region("prompt", prompt, VIS)]
        if self.anchor:
            regions.append(Region("anchor", prompt, HID))
        regions += [Region("live", live, LIVE), Region("target", target, LIVE)]
        return SeqLayout(regions)

    def prepare_condition(self, inputs) -> PreparedCondition:
        if isinstance(inputs, PreparedCondition):
            return inputs
        lay = self.layout(inputs["prompt"].shape[1], inputs["live"].shape[1], inputs["n_target"])
        return PreparedCondition(payload=dict(inputs), layout=lay)

    def _layout_of(self, condition) -> SeqLayout:
        """Derive this adapter's layout from the payload rather than using condition.layout.

        The teacher uses full attention and the student anchored attention, so condition.layout belongs to the adapter that prepared it and must not be reused blindly
        by another adapter.
        """
        return self.layout(
            condition["prompt"].shape[1], condition["live"].shape[1], condition["n_target"]
        )

    # 2. extract: compute the K/V for the VIS segment once
    def extract(self, condition, store=None) -> LayeredKV:
        lay = self._layout_of(condition)
        hidden = self._assemble(condition, self.init_state(condition))
        cache = LayeredKV(LAYERS, num_vis_tokens=lay.vis_len)
        plan = group_plan(lay, cached=False)
        self._run(hidden, lay, plan, cache=cache)
        return cache

    # 3. predict: consume reusable state
    def predict(self, state, step, condition, reusable=None) -> ModelOutput:
        lay = self._layout_of(condition)
        cached = reusable is not None
        plan = group_plan(lay, cached=cached)
        hidden = self._assemble(condition, state, live_only=cached)
        out = self._run(hidden, lay, plan, reusable=reusable)
        return ModelOutput(prediction=self.net.head(out["target"]) * float(step))

    def init_state(self, condition, **kwargs):
        n = condition["n_target"]
        return torch.zeros(1, n, DIM, dtype=condition["prompt"].dtype)

    def sample_train_state(self, batch):
        x0 = batch["x0"]
        sigma = batch["sigma"].view(-1, 1, 1)
        return (1 - sigma) * x0 + sigma * torch.randn_like(x0), batch["sigma"]

    def fold(self, pairs):
        return torch.cat([s for s, _ in pairs], 0), torch.cat([t for _, t in pairs], 0)

    # ── Private implementation details specific to this model; the framework is agnostic ──

    def _assemble(self, condition, state, live_only: bool = False):
        h = {"live": condition["live"], "target": state}
        if not live_only:
            h["prompt"] = condition["prompt"]
            if self.anchor:
                h["anchor"] = condition["prompt"]
        return h

    def _run(self, h, lay, plan, cache=None, reusable=None):
        h = dict(h)
        for i in range(LAYERS):
            proj = {n: self._proj(i, h[n]) for n in h}
            mq = torch.cat([proj[n][0] for n in plan.main_q], 1)
            mk = torch.cat([proj[n][1] for n in plan.main_kv if n in proj], 1)
            mv = torch.cat([proj[n][2] for n in plan.main_kv if n in proj], 1)
            if reusable is not None:
                kv = reusable.get(i)
                mk = torch.cat([mk, kv.key], 1)
                mv = torch.cat([mv, kv.value], 1)
            main = group_attn(mq, mk, mv, backend="sdpa")
            if cache is not None:
                vis = [r.name for r in lay.by_role(VIS)]
                cache.put(
                    i,
                    KVState(
                        torch.cat([proj[n][1] for n in vis], 1).clone(),
                        torch.cat([proj[n][2] for n in vis], 1).clone(),
                    ),
                )
            frozen = None
            if plan.has_frozen_group:
                frozen = group_attn(
                    torch.cat([proj[n][0] for n in plan.frozen_q], 1),
                    torch.cat([proj[n][1] for n in plan.frozen_kv], 1),
                    torch.cat([proj[n][2] for n in plan.frozen_kv], 1),
                    backend="sdpa",
                )
            self._writeback(h, i, plan, main, frozen)
        return h

    def _proj(self, i, x):
        b, s, _ = x.shape
        q, k, v = self.net.qkv[i](x).chunk(3, dim=-1)
        return tuple(t.reshape(b, s, HEADS, DIM // HEADS) for t in (q, k, v))

    def _writeback(self, h, i, plan, main, frozen):
        for names, val in ((plan.main_q, main), (plan.frozen_q, frozen)):
            if val is None:
                continue
            b, s, _, _ = val.shape
            merged = self.net.out[i](val.reshape(b, s, DIM))
            off = 0
            for n in names:
                ln = h[n].shape[1]
                h[n] = h[n] + merged[:, off : off + ln]
                off += ln


class ToySampler:
    def __init__(self, n=8):
        self.n = n

    def schedule(self, condition, **kwargs):
        return [1.0 - i / self.n for i in range(self.n)]

    def step(self, state, prediction, step, condition=None):
        return state + 0.1 * prediction


def main():
    torch.manual_seed(0)
    net = ToyBackend(seed=1)
    adapter = ToyAdapter(net, anchor=True)
    sampler = ToySampler(n=8)

    inputs = {
        "prompt": torch.randn(1, 6, DIM),
        "live": torch.randn(1, 4, DIM),
        "n_target": 5,
    }
    condition = adapter.prepare_condition(inputs)
    print("layout      :", condition.layout)
    print("degenerate  :", condition.layout.degenerate())
    print("mask shape  :", tuple(dense_mask_of(condition.layout).shape))

    # ── Inference ──
    reusable = adapter.extract(condition)
    print("cache layers:", len(reusable), "vis tokens:", reusable.num_vis_tokens)
    traj = rollout(adapter, condition, sampler, reusable=reusable)
    print("final state :", tuple(traj.final.shape))

    # ── extract invariance: changing state does not change the cache ──
    other = adapter.prepare_condition(dict(inputs))
    c2 = adapter.extract(other)
    same = all(
        torch.allclose(reusable.get(i).key, c2.get(i).key, atol=1e-6) for i in range(len(c2))
    )
    print("extract invariant:", same)

    # ── cached == extract ──
    state = adapter.init_state(condition)
    a = adapter.predict(state, 1.0, condition, reusable).prediction
    b = adapter.predict(state, 1.0, condition, None).prediction
    print("cached == full   :", torch.allclose(a, b, atol=1e-5), float((a - b).abs().max()))

    # ── runtime: changing the placement policy does not change numerical results ──
    mgr = CacheManager(residency=RollingResidency(offload_layers=1))
    mgr.begin_request("r", LAYERS)
    for i in range(LAYERS):
        mgr.put(_key(i), reusable.get(i))
    got = [mgr.get(_key(i)).key.clone() for i in range(LAYERS)]
    print("placement safe   :", all(torch.equal(got[i], reusable.get(i).key) for i in range(LAYERS)))
    print("runtime stats    :", {k: v for k, v in mgr.stats().items() if k != "pool"})

    # ── Training: switch providers between the two stages ──
    teacher = ToyAdapter(ToyBackend(seed=2), anchor=False)
    d = Distiller(teacher=teacher, student=adapter, match=mse, backward=lambda loss, last: None)
    batch = dict(inputs)
    batch.update(x0=torch.randn(1, 5, DIM), sigma=torch.rand(1))
    print("stage I     :", d.train_step(TeacherForced(num_states=2), batch))
    print("stage II    :", d.train_step(StudentRollout(sampler, num_states=3, lo=1), batch))

    # ── Guidance is a strategy object ──
    g = CfgGuidance(scale=1.0)
    print("guidance ok :", g.predict(adapter, state, 1.0, condition, reusable).prediction.shape)


def _key(layer: int):
    from anchorcache.core import CacheKey

    return CacheKey(request="r", layer=layer)


if __name__ == "__main__":
    main()
