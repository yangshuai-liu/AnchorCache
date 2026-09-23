from __future__ import annotations

from .config import TrainCfg

# Optimizer and lr scheduling. Plain AdamW: under ZeRO-2, DeepSpeed keeps an fp32 master copy and
# shards the state, so an 8-bit optimizer brings no benefit. A weights-only continuation fast-forwards
# the schedule start_step times to stay in phase with the full cosine schedule.


def build_optim(params, cfg: TrainCfg):
    import torch
    from diffusers.optimization import get_scheduler

    params = list(params)
    assert params, "no parameters with requires_grad"
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd, eps=cfg.eps)
    sched = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=cfg.warmup,
        num_training_steps=cfg.max_steps,
    )
    for _ in range(cfg.start_step):
        sched.step()
    return optim, sched
