"""Official DP-Forward release noise operator.

The public ``dp_noise.py`` calls ``matrix_gaussian_noise`` to obtain the
analytic Gaussian standard deviation and then adds i.i.d. normal entries
after clipping each sequence matrix to Frobenius norm ``norm_c``.  The name
is retained here for traceability.  The released inference implementation is
the diagonal/i.i.d. instantiation, not a learned full covariance matrix.
"""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Tuple

import torch


def _phi(value: float) -> float:
    return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0


def _log_phi_negative(value: float) -> float:
    """Return ``log(Phi(-value))`` without underflow for large ``value``.

    The DP-Forward release writes the analytic-Gaussian calibration as
    ``exp(epsilon) * Phi(-sqrt(...))``.  That expression is finite for the
    values used by the calibration, but its two factors respectively overflow
    and underflow when a deliberately very large epsilon is used as a
    near-noise-free control.  Work in log space in that tail instead.  The
    asymptotic Mills-ratio expansion is already accurate well before the
    branch point and avoids introducing a SciPy dependency.
    """

    if value < 0:
        raise ValueError("value must be non-negative")
    if value < 8.0:
        return math.log(0.5 * math.erfc(value / math.sqrt(2.0)))
    inverse_square = 1.0 / (value * value)
    # Phi(-x) = phi(x)/x * (1 - x^-2 + 3x^-4 - 15x^-6 + O(x^-8)).
    correction = 1.0 - inverse_square + 3.0 * inverse_square**2 - 15.0 * inverse_square**3
    return -0.5 * value * value - 0.5 * math.log(2.0 * math.pi) - math.log(value) + math.log(correction)


def _exp_epsilon_phi_negative(epsilon: float, value: float) -> float:
    """Compute ``exp(epsilon) * Phi(-value)`` stably.

    In the two release equations ``value = sqrt(epsilon * (u + 2))``, so the
    combined logarithm is bounded above even when ``epsilon`` itself is much
    larger than the range accepted by ``math.exp``.
    """

    log_value = epsilon + _log_phi_negative(value)
    # Values below this threshold round to zero in float64, which is the
    # correct numerical contribution for the bisection calibration.
    if log_value < -745.0:
        return 0.0
    return math.exp(log_value)


@lru_cache(maxsize=32)
def official_release_analytic_gaussian_std(
    epsilon: float, delta: float, sensitivity: float = 1.0, *, iterations: int = 5000
) -> float:
    """Port of DP-Forward release ``matrix_gaussian_noise`` exactly.

    This is the analytical Gaussian calibration used when the release command
    has ``--local_dp True --noise_type aGM``.  Its return value is passed as
    ``noise_factor`` to a coordinate-wise standard-normal draw.
    """

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("sensitivity must be finite and positive")

    def b_plus(value: float) -> float:
        return _phi(math.sqrt(epsilon * value)) - _exp_epsilon_phi_negative(
            epsilon, math.sqrt(epsilon * (value + 2.0))
        )

    def b_minus(value: float) -> float:
        return _phi(-math.sqrt(epsilon * value)) - _exp_epsilon_phi_negative(
            epsilon, math.sqrt(epsilon * (value + 2.0))
        )

    delta_zero = _phi(0.0) - _exp_epsilon_phi_negative(epsilon, math.sqrt(2.0 * epsilon))
    function = b_plus if delta >= delta_zero else b_minus
    start, end = 0.0, 1e5
    for _ in range(iterations):
        midpoint = (start + end) / 2.0
        if function(midpoint) < delta:
            end = midpoint
        else:
            start = midpoint
    u_star = end
    if delta >= delta_zero:
        alpha = math.sqrt(1.0 + u_star / 2.0) - math.sqrt(u_star / 2.0)
    else:
        alpha = math.sqrt(1.0 + u_star / 2.0) + math.sqrt(u_star / 2.0)
    calibration_r = math.sqrt(2.0 * epsilon) / alpha
    return float(sensitivity / calibration_r)


@dataclass(frozen=True)
class DPForwardConfig:
    epsilon: float
    delta: float = 1e-5
    norm_c: float = 1.0
    noise_layer: int = 28
    seed: int = 42

    def validate(self, total_layers: int) -> None:
        _ = official_release_analytic_gaussian_std(self.epsilon, self.delta)
        if not math.isfinite(self.norm_c) or self.norm_c <= 0:
            raise ValueError("norm_c must be finite and positive")
        if not 0 <= self.noise_layer < total_layers - 1:
            raise ValueError(
                f"noise_layer must be in [0, {total_layers - 2}], got {self.noise_layer}"
            )

    @property
    def noise_std(self) -> float:
        return official_release_analytic_gaussian_std(self.epsilon, self.delta)


def clip_and_add_official_release_noise(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    config: DPForwardConfig,
    sample_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply DP-Forward release clipping and Gaussian noise to active tokens.

    The original code reshapes each complete fixed-length BERT representation
    into a matrix and clips its Frobenius norm.  Padding is not semantic input
    to the LDP mechanism, so this LLaMA adaptation clips the corresponding
    active-token matrix instead.  Every active coordinate is then perturbed.
    """

    if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("Expected hidden_states [batch, sequence, hidden] and matching attention_mask")
    if sample_indices.numel() != hidden_states.shape[0]:
        raise ValueError("sample_indices must contain one id per batch element")
    output = hidden_states.clone()
    active_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    norm_sum = 0.0
    clipped_count = 0
    active_tokens = 0
    std = config.noise_std
    for row, index in enumerate(sample_indices.detach().cpu().tolist()):
        active = active_mask[row]
        values = hidden_states[row, active]
        count = int(values.shape[0])
        if count == 0:
            continue
        frobenius = values.float().norm(p=2)
        coefficient = min(1.0, float(config.norm_c / (frobenius.item() + 1e-6)))
        generator = torch.Generator(device=hidden_states.device)
        generator.manual_seed(int(config.seed + 1_000_003 * int(index)))
        noise = torch.randn(
            values.shape, device=hidden_states.device, dtype=torch.float32, generator=generator
        ).mul_(std).to(dtype=hidden_states.dtype)
        output[row, active] = values * coefficient + noise
        norm_sum += float(frobenius.item())
        clipped_count += int(coefficient < 1.0)
        active_tokens += count
    return output, {
        "active_tokens": float(active_tokens),
        "mean_pre_clip_frobenius": norm_sum / hidden_states.shape[0],
        "clipped_sequences": float(clipped_count),
        "noise_std": std,
    }
