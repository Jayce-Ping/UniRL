"""One-shot trainside native-vs-replay log-prob diagnostic (``UNIRL_DEBUG_RATIO``).

Wired into :meth:`FlowGRPO.compute_loss_and_backward` to fire **once per process**
on the first micro-batch of the first (on-policy) update — i.e. at **pre-update
weights** — when ``UNIRL_DEBUG_RATIO`` is set. It re-runs ``stage.replay`` under
three execution contexts and logs every signal needed to localize, in a single
run, why the on-policy importance ratio deviates from 1 on a *trainside*
(shared-engine) rollout.

Contexts (all on the SAME stored trajectory + SAME pre-update weights):

  - ``rA = (eval,  no_grad)`` — reproduces the rollout ``diffuse`` context.
  - ``rB = (train, no_grad)`` — isolates ``eval()`` → ``train()`` (dropout / mode).
  - ``rC = (train, grad)``    — reproduces the training ``replay`` context (== ``new_logp``).
  - ``rC2 = (train, grad)``   — a second ``rC`` to measure cross-call bf16/FSDP non-determinism.

Compared against the rollout-emitted native log-prob (``segment.sde_logp``):

  - ``Δlogp(native, rA)``  — SANITY. ~0 ⇒ a same-context replay reproduces the
    native sampling forward, so the gap is purely execution context (not a
    data/storage/path bug). Non-zero ⇒ inspect inputs first.
  - ``Δlogp(rB, rA)``      — ``eval()`` → ``train()`` contribution (dropout / mode).
  - ``Δlogp(rC, rB)``      — grad + activation-checkpointing contribution.
  - ``Δlogp(rC, native)``  — the ACTUAL on-policy gap; ``ratio = exp(·)``.
  - ``Δlogp(rC2, rC)``     — irreducible bf16/FSDP cross-call non-determinism floor.

It also reports the RAW per-step ``prev_sample_mean`` relative-L2 gap (the SDE
math is deterministic, so any mean gap is purely the transformer ``noise_pred``
forward divergence — with NO ``1/(2·std_var²)`` amplification), a ``Dropout(p>0)`` /
BatchNorm scan, and a full precision/schedule context dump.

All ranks run the replays (FSDP all-gather is collective); only rank 0 logs. No
side effects on training: the model's train/eval mode is restored on exit.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import gather_sde_field


def _rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _logp_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    """``|a-b|`` over ``[B, S']``: aggregate mean/max + per-step mean (over batch)."""
    d = (a.float() - b.float()).abs()
    return {
        "mean": float(d.mean()),
        "max": float(d.max()),
        "per_step": [float(d[:, s].mean()) for s in range(int(d.shape[1]))],
    }


def _mean_relgap(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    """Relative L2 gap of ``prev_sample_mean`` ``[B, S', *latent]`` per step (mean over batch).

    ``||a-b|| / ||b||`` over the latent dims. Since the SDE transition math is a
    deterministic function of the (identical) inputs, this isolates the transformer
    ``noise_pred`` forward divergence between the two contexts, unamplified.
    """
    if a is None or b is None:
        return None
    fa = a.float().flatten(2)
    fb = b.float().flatten(2)
    num = (fa - fb).norm(dim=2)
    den = fb.norm(dim=2).clamp_min(1e-12)
    rel = num / den
    return {
        "mean": float(rel.mean()),
        "max": float(rel.max()),
        "per_step": [float(rel[:, s].mean()) for s in range(int(rel.shape[1]))],
    }


def _transition_std(strategy: Any, sigma: float, sigma_next: float, eta: float, sigma_max: float) -> float:
    """Per-step transition Gaussian std for the amplification column.

    Prefers ``strategy.transition_std`` when present; otherwise replicates the
    Flow/Dance closed forms (``main`` has no ``transition_std`` method). Returns
    NaN for strategies it doesn't know — the column is diagnostic only.
    """
    if hasattr(strategy, "transition_std"):
        return float(
            strategy.transition_std(
                sigma=torch.tensor(sigma),
                sigma_next=torch.tensor(sigma_next),
                eta=eta,
                sigma_max=sigma_max,
            )
        )
    name = getattr(type(strategy), "canonical_name", "")
    neg_dt = max(sigma - sigma_next, 0.0)
    if name == "flow":
        denom = sigma_max if sigma == 1.0 else sigma
        if denom >= 1.0:
            return float("nan")
        return math.sqrt(sigma / (1.0 - denom)) * eta * math.sqrt(neg_dt)
    if name == "dance":
        return eta * math.sqrt(neg_dt)
    return float("nan")


def run_trainside_ratio_probe(
    *,
    stage: Any,
    conditions: Any,
    segment: Any,
    params: Any,
    target_steps: List[int],
    old_logp_source: str = "rollout",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Run the one-shot native-vs-replay attribution probe; see module docstring."""
    log = logger or logging.getLogger(__name__)
    rank = _rank()

    def emit(msg: str) -> None:
        if rank == 0:
            log.warning(msg)

    if not target_steps:
        emit("[DEBUG_RATIO] no SDE target steps on this segment; skipping probe.")
        return
    if segment.sde_logp is None or segment.sde_indices is None:
        emit("[DEBUG_RATIO] segment.sde_logp / sde_indices is None; cannot compare native vs replay.")
        return

    model = stage.trainable_module()
    prev_mode = bool(model.training)

    def replay_ctx(train_mode: bool, grad: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        model.train(train_mode)
        gctx = torch.enable_grad() if grad else torch.no_grad()
        with gctx:
            result = stage.replay(conditions, segment=segment, params=params, step_indices=target_steps)
        logp = result.log_probs.detach().float().cpu()
        means = result.prev_sample_means.detach().float().cpu() if result.prev_sample_means is not None else None
        del result  # drop any autograd graph from the grad contexts before the next replay
        return logp, means

    try:
        rA_logp, rA_mean = replay_ctx(False, False)  # eval,  no_grad == rollout `diffuse` ctx
        rB_logp, rB_mean = replay_ctx(True, False)   # train, no_grad
        rC_logp, rC_mean = replay_ctx(True, True)    # train, grad    == training `replay` ctx
        rC2_logp, rC2_mean = replay_ctx(True, True)  # determinism control
    finally:
        model.train(prev_mode)

    native = (
        gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp")
        .detach()
        .float()
        .cpu()
    )

    g_native_rA = _logp_stats(native, rA_logp)  # SANITY (~0)
    g_rB_rA = _logp_stats(rB_logp, rA_logp)      # eval -> train (mode/dropout)
    g_rC_rB = _logp_stats(rC_logp, rB_logp)      # grad + activation checkpointing
    g_rC_native = _logp_stats(rC_logp, native)   # ACTUAL on-policy gap
    g_floor = _logp_stats(rC2_logp, rC_logp)     # bf16/FSDP determinism floor

    m_rB_rA = _mean_relgap(rB_mean, rA_mean)
    m_rC_rB = _mean_relgap(rC_mean, rB_mean)
    m_rC_rA = _mean_relgap(rC_mean, rA_mean)
    m_floor = _mean_relgap(rC2_mean, rC_mean)

    ratio = torch.exp(rC_logp - native)
    ratio_std = float(ratio.std()) if ratio.numel() > 1 else 0.0
    approx_kl = float((0.5 * (rC_logp - native).pow(2)).mean())

    if rank != 0:
        return

    drops = [
        (n, float(getattr(mod, "p", 0.0)))
        for n, mod in model.named_modules()
        if isinstance(mod, torch.nn.Dropout) and float(getattr(mod, "p", 0.0)) > 0.0
    ]
    bns = [n for n, mod in model.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    sigmas = segment.sigmas.detach().float().cpu()
    eta = float(params.eta)
    sigma_max = float(sigmas[1]) if int(sigmas.shape[0]) > 1 else 0.99
    latent_shape = tuple(rA_mean.shape[2:]) if rA_mean is not None else None

    def fmt(m: Optional[Dict[str, Any]]) -> str:
        return "n/a" if m is None else f"mean={m['mean']:.3e} max={m['max']:.3e}"

    out: List[str] = []
    out.append("==================== [DEBUG_RATIO] trainside native-vs-replay probe ====================")
    out.append(
        f"  fired once @ pre-update weights | old_logp_source={old_logp_source!r} "
        f"| probe_batch={int(native.shape[0])} | sde_steps={target_steps}"
    )
    out.append(
        f"  model={type(model).__name__} training(orig)={prev_mode} "
        f"| dropout(p>0)={len(drops)} {drops[:6]} | batchnorm={len(bns)}"
    )
    out.append(
        f"  precisions: autocast={getattr(stage, 'autocast_dtype', None)} "
        f"trajectory={getattr(stage, 'trajectory_dtype', None)} logprob={getattr(stage, 'logprob_dtype', None)}"
    )
    out.append(
        f"  params: guidance_scale={getattr(params, 'guidance_scale', None)} eta={eta} "
        f"num_inference_steps={getattr(params, 'num_inference_steps', None)} "
        f"distilled_guidance_scale={getattr(params, 'distilled_guidance_scale', None)}"
    )
    out.append(
        f"  sde_indices={segment.sde_indices.tolist()} | sigma_max={sigma_max:.6f} "
        f"| latent(C,H,W)={latent_shape} | strategy={type(stage.strategy).__name__}"
    )
    bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a"
    out.append(f"  torch={torch.__version__} cuda={torch.version.cuda} bf16_supported={bf16}")
    out.append("  ---- AGGREGATE |Δlogp| (mean over batch×step) --------------------------------------")
    out.append(
        f"    SANITY   native vs rA(eval,no_grad) = {g_native_rA['mean']:.3e} (max {g_native_rA['max']:.3e})"
        "   [≈0 ⇒ same-ctx replay reproduces native]"
    )
    out.append(
        f"    MODE     rB(train) - rA(eval)       = {g_rB_rA['mean']:.3e} (max {g_rB_rA['max']:.3e})"
        "   [eval()->train(): dropout/mode]"
    )
    out.append(
        f"    GRAD/AC  rC(grad) - rB(no_grad)     = {g_rC_rB['mean']:.3e} (max {g_rC_rB['max']:.3e})"
        "   [grad + activation checkpointing]"
    )
    out.append(
        f"    FLOOR    rC2 - rC (same ctx)        = {g_floor['mean']:.3e} (max {g_floor['max']:.3e})"
        "   [bf16/FSDP cross-call non-determinism]"
    )
    out.append(
        f"    >>TOTAL  rC(train,grad) - native    = {g_rC_native['mean']:.3e} (max {g_rC_native['max']:.3e})"
        "   [the on-policy gap]"
    )
    out.append("  ---- on-policy RATIO = exp(rC - native) --------------------------------------------")
    out.append(
        f"    mean={float(ratio.mean()):.6f} std={ratio_std:.6f} "
        f"min={float(ratio.min()):.6f} max={float(ratio.max()):.6f} | approx_kl={approx_kl:.3e}"
    )
    out.append(f"    abs logp scale: native.mean={float(native.mean()):.4f} rC.mean={float(rC_logp.mean()):.4f}")
    nonfinite = [
        name
        for name, t in (("native", native), ("rA", rA_logp), ("rB", rB_logp), ("rC", rC_logp))
        if not bool(torch.isfinite(t).all())
    ]
    if nonfinite:
        out.append(f"    !! WARNING: non-finite logp in {nonfinite} — gaps/ratio above are unreliable.")
    out.append("  ---- RAW prev_sample_mean rel-L2 gap (no 1/(2σ²) amp ⇒ transformer forward diff) ----")
    out.append(f"    MODE    rB-rA  = {fmt(m_rB_rA)}")
    out.append(f"    GRAD/AC rC-rB  = {fmt(m_rC_rB)}")
    out.append(f"    TOTAL   rC-rA  = {fmt(m_rC_rA)}")
    out.append(f"    FLOOR   rC2-rC = {fmt(m_floor)}")
    out.append("  ---- per-step (σ, std_var, mean-relgap rC-rA, |Δlogp| rC-native, ratio, amp) --------")
    relgap_ps = m_rC_rA["per_step"] if m_rC_rA is not None else [float("nan")] * len(target_steps)
    for j, step in enumerate(target_steps):
        s = float(sigmas[step])
        s_next = float(sigmas[step + 1])
        std_var = _transition_std(stage.strategy, s, s_next, eta, sigma_max)
        rj = torch.exp(rC_logp[:, j] - native[:, j])
        amp = relgap_ps[j] / std_var if (std_var == std_var and std_var > 0) else float("nan")
        out.append(
            f"    step {step:>3}: σ={s:.4f}->{s_next:.4f} std_var={std_var:.4f} "
            f"san={g_native_rA['per_step'][j]:.2e} mean_relgap={relgap_ps[j]:.3e} "
            f"|Δlogp|={g_rC_native['per_step'][j]:.3e} "
            f"ratio[mean={float(rj.mean()):.5f} max={float(rj.max()):.5f}] amp={amp:.3e}"
        )
    out.append("  ---- HINT --------------------------------------------------------------------------")
    if g_native_rA["mean"] > 1e-4:
        out.append(
            "    native vs rA is NOT ~0: the rollout `diffuse` forward differs from a same-context "
            "replay — inspect x_t / sigma / conditions / trajectory storage before anything else."
        )
    contrib = {"MODE(eval->train)": g_rB_rA["mean"], "GRAD/AC": g_rC_rB["mean"], "FLOOR(bf16/FSDP)": g_floor["mean"]}
    dominant = max(contrib, key=contrib.get)
    out.append(f"    dominant contributor to on-policy gap: {dominant} ({contrib[dominant]:.3e})")
    if drops:
        out.append(
            f"    dropout(p>0) present ({len(drops)}): eval()->train() diverges — set dropout=0 so native aligns."
        )
    out.append("    MODE dominant            ⇒ remove eval/train (dropout) gap; native (rollout) source aligns.")
    out.append("    FLOOR≈TOTAL (irreducible) ⇒ old_logp_source=replay is the principled fix (exact ratio=1).")
    out.append("========================================================================================")
    log.warning("\n".join(out))
