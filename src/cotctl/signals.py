"""Per-step state signals, computed from raw pre-temperature logits.

Sensor/actuator separation is the load-bearing design decision of the whole
project and it lives entirely in this module:

    * Every signal below is computed from the RAW logits z_t at the reference
      temperature T_ref = 1.0. None of them depends on the sampling temperature.
    * The sampling temperature acts only inside `decoding.sample_token`, i.e. on
      the actuator side, never on the sensor side.

If the measured output were the entropy of the TEMPERED distribution, the
control input would appear algebraically in the measurement (direct
feedthrough, dH/dT = Var_p(z) / T**3 > 0, monotone and analytically
invertible), the loop would degenerate into a static inversion, and the whole
thing would reduce to Mirostat with a PID bolted on. Do not "simplify" this.

Var_p(z) is exported here for convenience, but it is NOT a state feature: it is
the actuator gain (dH/dT at T = 1). It belongs in gain scheduling, not in the
state vector.

Entropy is in nats throughout.
"""

from __future__ import annotations

from typing import Final

import torch

REF_TEMPERATURE: Final[float] = 1.0

# Order MUST match the tensor returned by `compute_step_signals`. The signals are
# kept as GPU tensors and stacked once per trajectory rather than being read out
# with .item() every step: a per-signal readout forces a host sync on every token
# and dominates the wall clock at 1.5B scale.
# Temperatures at which H is evaluated per step, spanning and overshooting the
# admissible actuator band [0.5, 1.5]. Two points gave only the band edges, which
# is enough to ask "is the target reachable" and not enough to ask "what
# temperature would reach it" -- the first diagnostic run had to answer the second
# question by linearising dH/dT at T = 1, which is crude far from T = 1.
#
# Cost measured directly, with warm-up and alternating order because the first
# run of anything on this card pays for CUDA warm-up and would otherwise report
# the second configuration as faster: 12.16 tok/s at three points against 11.82
# at seven, so the four extra softmaxes cost 2.8% of throughput. Small, but not
# the "noise" first claimed -- an isolated benchmark of the sensor understates
# its marginal cost, because in a launch-bound loop the CPU issue rate is the
# constraint, not GPU time. The grid is recorded in the run config, so a run
# stays self-describing if it ever changes.
TEMPERATURE_GRID: Final[tuple[float, ...]] = (0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0)

# Edges of the admissible band, kept as named columns as well as grid points:
# they mean something the other grid points do not, namely the actuator's real
# limits, and keeping them preserves comparability with runs already written.
T_MIN_INDEX: Final[int] = TEMPERATURE_GRID.index(0.5)
T_MAX_INDEX: Final[int] = TEMPERATURE_GRID.index(1.5)
T_REF_INDEX: Final[int] = TEMPERATURE_GRID.index(1.0)

BASE_SIGNAL_NAMES: Final[tuple[str, ...]] = (
    "entropy",           # H^(1)_t
    "margin",            # p_(1) - p_(2)
    "surprisal",         # -log p^(1)[sampled token]
    "kl_prev",           # KL(p_t || p_{t-1}); NaN at t = 0
    "logit_var",         # Var_p(z) = actuator gain dH/dT at T = 1 (NOT a state feature)
)
GRID_SIGNAL_NAMES: Final[tuple[str, ...]] = tuple(
    f"entropy_at_T{t:g}" for t in TEMPERATURE_GRID
)
SIGNAL_NAMES: Final[tuple[str, ...]] = BASE_SIGNAL_NAMES + GRID_SIGNAL_NAMES
N_SIGNALS: Final[int] = len(SIGNAL_NAMES)
N_BASE_SIGNALS: Final[int] = len(BASE_SIGNAL_NAMES)


def tempered_entropy(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Shannon entropy of softmax(z / T), in nats.

    Used both for the reference measurement (T = 1) and for the two edges of the
    controllability map (T_min, T_max). Logits must already be float32.
    """
    log_p = torch.log_softmax(logits / temperature, dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1)


@torch.no_grad()
def compute_step_signals(
    logits: torch.Tensor,
    sampled_token: torch.Tensor,
    prev_log_probs: torch.Tensor | None,
    grid: tuple[float, ...] = TEMPERATURE_GRID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the state signals for one decoding step.

    Args:
        logits: Raw logits, shape (V,), float32. Upcasting from bf16 is the
            caller's job and is mandatory: bf16 has ~8 mantissa bits, which puts
            a ~0.4% relative floor on the logits and up to ~8% error on tail
            probabilities.
        sampled_token: 0-dim int tensor, the token actually drawn at this step.
            Note this is the token drawn at the SAMPLING temperature, but its
            surprisal is scored under the reference distribution.
        prev_log_probs: log p_{t-1} over the full vocabulary at T_ref, shape (V,),
            or None at the first step.
        grid: Temperatures at which to evaluate H. Must contain 1.0, 0.5 and
            1.5; defaults to TEMPERATURE_GRID.

    Returns:
        (signals, log_probs) where `signals` is a (N_SIGNALS,) float32 tensor
        ordered as SIGNAL_NAMES, and `log_probs` is log p_t at T_ref, to be fed
        back in as `prev_log_probs` on the next step.
    """
    log_p = torch.log_softmax(logits, dim=-1)  # T_ref = 1.0 by construction
    p = log_p.exp()

    entropy = -(p * log_p).sum()

    top2 = torch.topk(p, k=2)
    margin = top2.values[0] - top2.values[1]

    surprisal = -log_p[sampled_token]

    if prev_log_probs is None:
        kl_prev = torch.full((), float("nan"), device=logits.device)
    else:
        kl_prev = (p * (log_p - prev_log_probs)).sum()

    mean_z = (p * logits).sum()
    logit_var = (p * logits.square()).sum() - mean_z.square()

    signals = torch.stack(
        [entropy, margin, surprisal, kl_prev, logit_var]
        + [tempered_entropy(logits, t) for t in grid]
    )
    return signals, log_p


@torch.no_grad()
def compute_step_signals_batched(
    logits: torch.Tensor,
    sampled_tokens: torch.Tensor,
    prev_log_probs: torch.Tensor | None,
    grid: tuple[float, ...] = TEMPERATURE_GRID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched twin of `compute_step_signals`. Same maths, one launch per op.

    This exists rather than looping the single-sequence version because looping
    would multiply the sensor's kernel launches by the batch size, and kernel
    launches are precisely what batching exists to amortise: the loop is
    CPU-launch-bound, not bandwidth-bound. At B = 16 a per-sequence sensor would
    cost ~22 ms against a ~72 ms forward instead of ~1.4 ms.

    The single-sequence version is deliberately left untouched. It is the
    reference the batching regression test compares against, and a reference that
    shares an implementation with the thing under test proves nothing.

    Args:
        logits: (B, V) raw logits, float32.
        sampled_tokens: (B,) tokens actually drawn this step.
        prev_log_probs: (B, V) log p_{t-1} at T_ref, or None at the first step.
        grid: temperatures at which to evaluate H.

    Returns:
        ((B, N_SIGNALS) float32, (B, V) log-probs at T_ref).
    """
    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()

    entropy = -(p * log_p).sum(dim=-1)

    top2 = torch.topk(p, k=2, dim=-1)
    margin = top2.values[:, 0] - top2.values[:, 1]

    surprisal = -log_p.gather(1, sampled_tokens.unsqueeze(1)).squeeze(1)

    if prev_log_probs is None:
        kl_prev = torch.full_like(entropy, float("nan"))
    else:
        kl_prev = (p * (log_p - prev_log_probs)).sum(dim=-1)

    mean_z = (p * logits).sum(dim=-1)
    logit_var = (p * logits.square()).sum(dim=-1) - mean_z.square()

    signals = torch.stack(
        [entropy, margin, surprisal, kl_prev, logit_var]
        + [tempered_entropy(logits, t) for t in grid],
        dim=-1,
    )
    return signals, log_p


def count_reachable_band_violations(
    signals: torch.Tensor,
    tolerance: float = 1e-4,
    grid: tuple[float, ...] = TEMPERATURE_GRID,
) -> int:
    """Steps where H(T) fails to increase monotonically across the grid.

    Because dH/dT = Var_p(z)/T**3 > 0, entropy is strictly increasing in T, so
    the grid must be non-decreasing at every step. With two points this could
    only check that H(T_ref) sat between the band edges; across seven points it
    checks the shape of the whole curve, and catches a swapped bound, an
    inverted sign or a mis-ordered grid in one run.

    Checked once per trajectory on the stacked CPU tensor rather than per step:
    a per-step assertion would force a host sync on every token and dominate the
    wall clock. `tolerance` absorbs float32 rounding at points that are genuinely
    equal, which happens when the distribution is uniform.

    Args:
        signals: (n_steps, N_SIGNALS), ordered as SIGNAL_NAMES.
        tolerance: absorbs float32 rounding at genuinely equal points.
        grid: the grid the signals were computed with. Must be passed when it is
            not the default -- the index of T = 1 depends on it, and reading that
            index from the module constant would silently address the wrong
            column, or raise, on any other grid.

    Returns:
        Number of steps with at least one violation; 0 is the only acceptable value.
    """
    curve = signals[:, N_BASE_SIGNALS:]
    if curve.shape[1] != len(grid):
        raise ValueError(
            f"signals carry {curve.shape[1]} grid points but grid has {len(grid)}"
        )

    decreasing = curve[:, 1:] < curve[:, :-1] - tolerance

    # The reference entropy must also agree with the grid point at T = 1.
    reference = signals[:, SIGNAL_NAMES.index("entropy")]
    mismatch = (reference - curve[:, grid.index(REF_TEMPERATURE)]).abs() > tolerance

    return int((decreasing.any(dim=1) | mismatch).sum())


def entropy_derivative_wrt_temperature(
    logit_var: torch.Tensor, temperature: float
) -> torch.Tensor:
    """dH/dT = Var_p(z) / T**3, with p = softmax(z / T).

    Derivation: with beta = 1/T and p_i proportional to exp(beta * z_i), we have
    dH/dbeta = -beta * Var_p(z) and dbeta/dT = -1/T**2, hence
    dH/dT = beta * Var_p(z) / T**2 = Var_p(z) / T**3 > 0.

    This is the plant gain of the NAIVE (tempered-entropy) formulation, and its
    closed form is precisely why that formulation is a static inversion rather
    than feedback. It is kept here because it is the negative result the
    conference thesis reports, and because the same Var_p(z) is the gain that
    the real controller must schedule against.

    Note that for the SENSOR used here, dH^(1)/dT is identically zero by
    construction; the true plant gain must be identified empirically
    (step/PRBS response -> FOPDT -> IMC tuning).
    """
    return logit_var / temperature**3
