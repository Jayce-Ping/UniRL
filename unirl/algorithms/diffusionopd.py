"""DiffusionOPD — on-policy distillation for diffusion models (teacher-anchored)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.algorithms.base import (
    AlgorithmStepResult,
    StageAlgorithm,
    _gaussian_kl_div,
    _transition_sigma,
    gather_sde_field,
    typed_conditions,
)
from unirl.train.lora import adapter_active, adapter_names

DOMAIN_KEY = "domain"
_LOSS_TARGETS = ("xt", "v", "x0")
_SELF_NORMALIZE_EPS = 1e-8


@dataclass
class TeacherSpec:
    """One distillation teacher."""

    name: str
    guidance_scale: Optional[float] = None


def _project_velocity_target(
    *,
    loss_target: str,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """Project stacked ``(x_t, v)`` into the ``v`` or ``x0`` distillation space."""
    if loss_target == "v":
        return velocity
    if loss_target != "x0":
        raise ValueError(
            f"DiffusionOPD velocity projection expects loss_target 'v' or 'x0', got {loss_target!r}."
        )
    if velocity.shape != latents.shape:
        raise ValueError(
            "DiffusionOPD x0 projection requires matching latents and velocity shapes, "
            f"got latents={tuple(latents.shape)} and velocity={tuple(velocity.shape)}."
        )
    if sigmas.ndim != 1 or int(sigmas.shape[0]) != int(latents.shape[1]):
        raise ValueError(
            "DiffusionOPD x0 projection expects sigmas shape [S'] matching latents dim-1, "
            f"got sigmas={tuple(sigmas.shape)} and latents={tuple(latents.shape)}."
        )
    sigma = sigmas.to(device=latents.device, dtype=torch.float32).view(
        1, latents.shape[1], *([1] * (latents.ndim - 2))
    )
    return latents.float() - sigma * velocity.float()


def _align_velocity_to_sample(velocity: torch.Tensor, sample: torch.Tensor, *, step_idx: int) -> torch.Tensor:
    """Match ``predict_noise_at_step`` output to ``latents_at`` (Bagel may drop a leading 1)."""
    if velocity.shape == sample.shape:
        return velocity
    if velocity.ndim == sample.ndim - 1 and sample.shape[0] == 1 and velocity.shape == sample.shape[1:]:
        return velocity.unsqueeze(0)
    raise ValueError(
        f"DiffusionOPD: predict_noise_at_step at step_idx={step_idx} returned {tuple(velocity.shape)}, "
        f"expected {tuple(sample.shape)} to match latents_at({step_idx})."
    )


def _reduce_distill_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    kl_sigma: Optional[torch.Tensor],
    self_normalize: bool,
) -> torch.Tensor:
    """Mean over spatial dims then (B, S'). ``kl_sigma`` set ⇒ Gaussian KL; else plain MSE."""
    if student.shape != teacher.shape:
        raise ValueError(
            "DiffusionOPD loss requires matching student/teacher target shapes, "
            f"got student={tuple(student.shape)} and teacher={tuple(teacher.shape)}."
        )
    student_f32 = student.float()
    teacher_f32 = teacher.float()
    error = student_f32 - teacher_f32
    if kl_sigma is None:
        per_elem = error**2
    else:
        per_elem = _gaussian_kl_div(student_f32, teacher_f32, kl_sigma)
    spatial = tuple(range(2, per_elem.ndim))
    per_sample_step = per_elem.mean(dim=spatial) if spatial else per_elem
    if self_normalize:
        mae = error.abs().mean(dim=spatial).detach() if spatial else error.abs().detach()
        per_sample_step = per_sample_step / (mae + _SELF_NORMALIZE_EPS)
    return per_sample_step.mean()


class DiffusionOPD(StageAlgorithm):
    """Teacher-anchored distillation loss over the student's own rollout."""

    requires_ema_rollout = False
    # Multi-update against the teacher anchor is unvalidated.
    supports_multi_update = False
    # Teacher adapters live on the trainable model; the trainer injects the backend.
    requires_backend = True
    # Supervision is teacher-driven; rewards (if any) are monitoring-only.
    requires_advantages = False

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
        teachers: Any = None,
        add_kl_coefficient: bool = False,
        loss_target: str = "xt",
        self_normalize: bool = False,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is not None:
            stage = getattr(pipeline, stage_attr)
        if stage is None:
            raise ValueError("DiffusionOPD: either `stage` or `pipeline` must be provided")
        self.stage = stage
        self.params = params
        self.conditions_cls = conditions_cls
        self.add_kl_coefficient = bool(add_kl_coefficient)
        if not isinstance(loss_target, str):
            raise TypeError(
                f"DiffusionOPD: expected str for loss_target, got {type(loss_target).__name__}: {loss_target!r}"
            )
        if loss_target not in _LOSS_TARGETS:
            raise ValueError(
                f"DiffusionOPD: loss_target must be one of {_LOSS_TARGETS}, got {loss_target!r}."
            )
        if not isinstance(self_normalize, bool):
            raise TypeError(
                "DiffusionOPD: expected bool for self_normalize, "
                f"got {type(self_normalize).__name__}: {self_normalize!r}"
            )
        self.loss_target = loss_target
        self.self_normalize = self_normalize
        if self.add_kl_coefficient and not float(getattr(params, "eta", 0.0)) > 0.0:
            raise ValueError(
                "DiffusionOPD: add_kl_coefficient=True normalizes by the SDE transition std, "
                f"which scales with sampling eta; got eta={getattr(params, 'eta', None)!r}. "
                "Use a noised rollout (eta > 0), or add_kl_coefficient=False for ODE mean-matching."
            )
        if self.loss_target in ("v", "x0") and self.add_kl_coefficient:
            raise ValueError(
                "DiffusionOPD: add_kl_coefficient=True is only defined for loss_target='xt' "
                f"(Gaussian KL on transition means); got loss_target={self.loss_target!r}."
            )
        if self.loss_target in ("v", "x0") and not callable(getattr(stage, "predict_noise_at_step", None)):
            raise TypeError(
                f"DiffusionOPD: loss_target={self.loss_target!r} needs stage.predict_noise_at_step; "
                f"{type(stage).__name__} does not provide it."
            )

        self.teachers: Dict[str, TeacherSpec] = {}
        for entry in teachers or []:
            if isinstance(entry, TeacherSpec):
                spec = entry
            else:
                get = entry.get if hasattr(entry, "get") else lambda k, d=None: getattr(entry, k, d)
                name = get("name")
                if not name:
                    raise ValueError(f"DiffusionOPD: every teacher entry needs a 'name'; got {entry!r}.")
                gs = get("guidance_scale")
                spec = TeacherSpec(name=str(name), guidance_scale=None if gs is None else float(gs))
            if spec.name in self.teachers:
                raise ValueError(f"DiffusionOPD: duplicate teacher name {spec.name!r}.")
            self.teachers[spec.name] = spec
        if not self.teachers:
            raise ValueError("DiffusionOPD: at least one teacher is required.")

        model = getattr(backend, "model", None) if backend is not None else None
        if model is None:
            raise ValueError(
                "DiffusionOPD: no `backend` was injected — the teacher adapters live on the "
                "trainable model. The v2 DiffusionTrainer injects it when the algorithm "
                "declares requires_backend=True."
            )
        self._model = model
        present = adapter_names(model)
        missing = sorted(set(self.teachers) - present)
        if missing:
            raise ValueError(
                f"DiffusionOPD: teacher adapter(s) {missing} not found on the trainable model "
                f"(present: {sorted(present)}). Declare them under "
                "backend.lora_cfg.frozen_adapters as {name, path} entries."
            )
        # Set by prepare_part for the per-teacher loss metric of the current rollout.
        self._active_teacher: Optional[str] = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _resolve_target_steps(self, segment: Any) -> List[int]:
        """All SDE-recorded step indices on the segment (mirrors FlowDPPO)."""
        if segment is None or segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]

    def _resolve_teacher(self, part: Any) -> TeacherSpec:
        """Single-domain teacher for this train Part."""
        domains = {(md or {}).get(DOMAIN_KEY) for md in (part.metadata or [None] * part.batch_size)}
        if domains == {None}:
            raise RuntimeError(
                "DiffusionOPD.prepare_part: no per-row metadata[{key!r}] on the train Part. "
                "Use a domain-stamping data source (MultiDomainRLDataSource) — the trainer "
                "projects root metadata onto the train Part automatically.".format(key=DOMAIN_KEY)
            )
        if len(domains) != 1:
            raise RuntimeError(
                f"DiffusionOPD.prepare_part: mixed-domain batch {sorted(str(d) for d in domains)}; "
                "each rollout batch must be single-domain (one teacher per batch)."
            )
        domain = str(next(iter(domains)))
        teacher = self.teachers.get(domain)
        if teacher is None:
            raise RuntimeError(
                f"DiffusionOPD.prepare_part: batch domain {domain!r} has no configured teacher "
                f"(teachers: {sorted(self.teachers)})."
            )
        return teacher

    def _teacher_params(self, teacher: TeacherSpec) -> Any:
        if teacher.guidance_scale is None:
            return self.params
        return dataclasses.replace(self.params, guidance_scale=teacher.guidance_scale)

    def _replay_means(self, conditions: Any, segment: Any, params: Any, target_steps: List[int]) -> torch.Tensor:
        result = self.stage.replay(
            conditions,
            segment=segment,
            params=params,
            step_indices=target_steps,
        )
        if result.prev_sample_means is None:
            raise RuntimeError(
                "DiffusionOPD: stage.replay() returned prev_sample_means=None; "
                "loss_target='xt' requires the stage's replay to produce means."
            )
        return result.prev_sample_means

    def _velocity_targets(self, conditions: Any, segment: Any, params: Any, target_steps: List[int]) -> torch.Tensor:
        """Per-step ``predict_noise_at_step`` stacked to ``[B, S', *latent]``, then projected."""
        if segment.sigmas is None:
            raise RuntimeError(
                f"DiffusionOPD: segment.sigmas is None (needed for loss_target={self.loss_target!r})."
            )
        velocities: List[torch.Tensor] = []
        latents: List[torch.Tensor] = []
        sigmas: List[torch.Tensor] = []
        for step_idx in target_steps:
            sample = segment.latents_at(step_idx)
            sigma = segment.sigmas[int(step_idx)]
            velocity = self.stage.predict_noise_at_step(
                conditions, sample=sample, sigma=sigma, params=params
            )
            velocity = _align_velocity_to_sample(velocity, sample, step_idx=step_idx)
            velocities.append(velocity)
            latents.append(sample.to(device=velocity.device))
            sigmas.append(torch.as_tensor(sigma, device=velocity.device, dtype=torch.float32).reshape(()))
        return _project_velocity_target(
            loss_target=self.loss_target,
            latents=torch.stack(latents, dim=1),
            sigmas=torch.stack(sigmas, dim=0),
            velocity=torch.stack(velocities, dim=1),
        )

    def _student_target(self, conditions: Any, segment: Any, target_steps: List[int]) -> torch.Tensor:
        if self.loss_target == "xt":
            return self._replay_means(conditions, segment, self.params, target_steps)
        return self._velocity_targets(conditions, segment, self.params, target_steps)

    def prepare_part(self, part: Any) -> Any:
        """Freeze the batch's teacher target on ``segment.sde_means`` (μ / v / x0)."""
        target_steps = self._resolve_target_steps(part.segment)
        if not target_steps:
            return part

        teacher = self._resolve_teacher(part)
        teacher_params = self._teacher_params(teacher)
        typed_conds = typed_conditions(part.conditions, self.conditions_cls)
        with torch.no_grad(), adapter_active(self._model, teacher.name):
            if self.loss_target == "xt":
                target = self._replay_means(typed_conds, part.segment, teacher_params, target_steps)
            else:
                target = self._velocity_targets(typed_conds, part.segment, teacher_params, target_steps)
        part.segment.sde_means = target.detach().cpu()
        self._active_teacher = teacher.name
        return part

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Any],
        segment: Any,
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """Per-step student vs frozen teacher target (KL on ``xt``, MSE on ``v``/``x0``)."""
        target_steps = self._resolve_target_steps(segment)
        if not target_steps or segment.sde_means is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        student = self._student_target(typed_conds, segment, target_steps)
        teacher = gather_sde_field(segment.sde_means, segment.sde_indices, target_steps, field_name="sde_means")
        teacher = teacher.to(device=student.device)

        kl_sigma: Optional[torch.Tensor] = None
        if self.loss_target == "xt":
            kl_sigma = _transition_sigma(
                self.stage,
                segment=segment,
                target_steps=target_steps,
                eta=float(getattr(self.params, "eta", 1.0)),
                device=student.device,
                add_coefficient=self.add_kl_coefficient,
            )
        loss = _reduce_distill_loss(
            student,
            teacher,
            kl_sigma=kl_sigma,
            self_normalize=self.self_normalize,
        )

        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {"distill_loss": float(loss.detach().item())}
        if self._active_teacher is not None:
            # Only this rollout's domain emits the key -> one wandb series per teacher.
            metrics[f"distill_loss_{self._active_teacher}"] = metrics["distill_loss"]
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )


__all__ = ["DiffusionOPD", "TeacherSpec", "DOMAIN_KEY"]
