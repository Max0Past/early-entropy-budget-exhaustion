#!/usr/bin/env python
"""Environment smoke test: calibrate the instrument before trusting a measurement.

Run this after building the environment, after any change to `signals.py`, and on
any new machine. It answers one question: is the sensor measuring what we think
it is measuring?

The checks are ordered so that the ones with an exact analytic answer come first.
Uniform logits must give H = ln(V) and Var_p(z) = 0 -- these are the "known weight
on the scale". A sensor that fails them is wrong in a way that would otherwise
never crash: the run would finish, write a well-formed Parquet, and produce plots
made of nonsense.

No model weights are downloaded. Every check runs on synthetic logits at the real
Qwen vocabulary width, so this costs seconds.

Usage:
    python scripts/check_env.py

Exits 0 if every check passes, 1 otherwise, so it is safe to run automatically.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path
from typing import Final

# The package is not installed yet (no pyproject.toml), so make `src/` importable.
# Remove this once `pip install -e .` is the way in.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from cotctl.signals import (  # noqa: E402
    GRID_SIGNAL_NAMES,
    SIGNAL_NAMES,
    compute_step_signals,
    count_reachable_band_violations,
    entropy_derivative_wrt_temperature,
    tempered_entropy,
)

# Qwen2 / Qwen3 vocabulary. The maths below is width-agnostic; using the real
# width keeps the numbers representative of an actual decoding step.
VOCAB_SIZE: Final[int] = 151_936

# Admissible temperature band, matching DecodeConfig defaults.
T_MIN: Final[float] = 0.5
T_MAX: Final[float] = 1.5

# Invariant 8: these do not move until Phase I is finished. A silent upgrade does
# not raise an error, it just makes runs incomparable -- which is why it is checked.
EXPECTED_TRANSFORMERS: Final[str] = "4.57.1"
EXPECTED_TORCH: Final[str] = "2.11.0+cu128"

# bf16 keeps ~8 mantissa bits. Rounding logits through it perturbs the entropy;
# the perturbation is the sensor's noise floor and must be non-zero (otherwise the
# fp32 upcast in decoding.py would be pointless) but small.
BF16_FLOOR_MIN: Final[float] = 1e-6


class Report:
    """Collects check results and decides the exit code."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, label: str, passed: bool, detail: str = "") -> None:
        if not passed:
            self.failures.append(label)
        status = "PASS" if passed else "FAIL"
        suffix = f"  {detail}" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    def info(self, label: str, value: str) -> None:
        print(f"         {label}: {value}")

    def summarise(self) -> int:
        print()
        if self.failures:
            print(f"RESULT: {len(self.failures)} CHECK(S) FAILED")
            for label in self.failures:
                print(f"  - {label}")
            return 1
        print("RESULT: ALL CHECKS PASSED")
        return 0


def check_versions(report: Report) -> None:
    """Pinned versions. An unnoticed upgrade silently invalidates comparability."""
    print("=== 1. pinned versions (invariant 8) ===")
    import transformers

    report.check(
        f"transformers == {EXPECTED_TRANSFORMERS}",
        transformers.__version__ == EXPECTED_TRANSFORMERS,
        f"found {transformers.__version__}",
    )
    report.check(
        f"torch == {EXPECTED_TORCH}",
        torch.__version__ == EXPECTED_TORCH,
        f"found {torch.__version__}",
    )


def check_device(report: Report) -> str:
    """GPU visibility and bf16 support. Returns the device to run the sensor on."""
    print("=== 2. device ===")
    available = torch.cuda.is_available()
    report.check("CUDA available", available)
    if not available:
        report.info("note", "sensor checks will run on CPU; GPU results unverified")
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info()
    report.info("device", torch.cuda.get_device_name(0))
    report.info("capability", f"sm_{major}{minor}")
    report.info("vram free/total", f"{free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    report.check("bf16 supported natively", torch.cuda.is_bf16_supported())
    return "cuda"


def check_sensor_ground_truth(report: Report, device: str) -> None:
    """Uniform logits have an exact closed-form answer. This is the known weight."""
    print("=== 3. sensor vs analytic ground truth (uniform logits) ===")
    logits = torch.zeros(VOCAB_SIZE, device=device, dtype=torch.float32)
    token = torch.tensor(0, device=device)

    signals, _ = compute_step_signals(logits, token, None)
    values = dict(zip(SIGNAL_NAMES, signals.tolist()))
    ln_v = math.log(VOCAB_SIZE)

    report.check("H == ln(V)", abs(values["entropy"] - ln_v) < 1e-3,
                 f"{values['entropy']:.6f} vs {ln_v:.6f}")
    report.check("margin == 0", abs(values["margin"]) < 1e-6,
                 f"{values['margin']:.3e}")
    report.check("Var_p(z) == 0", abs(values["logit_var"]) < 1e-6,
                 f"{values['logit_var']:.3e}")
    report.check("surprisal == ln(V)", abs(values["surprisal"] - ln_v) < 1e-3,
                 f"{values['surprisal']:.6f}")
    report.check("kl_prev is NaN at t = 0", math.isnan(values["kl_prev"]))


def _peaked_logits(device: str) -> torch.Tensor:
    """A distribution shaped like a real next-token step: broad, with one clear mode."""
    generator = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(VOCAB_SIZE, generator=generator, device=device,
                         dtype=torch.float32) * 2.0
    logits[42] += 8.0
    return logits


def check_reachable_band(report: Report, device: str) -> None:
    """The two controllability-map columns must bracket the reference measurement."""
    print("=== 4. reachable entropy band (controllability map columns) ===")
    logits = _peaked_logits(device)
    token = torch.tensor(42, device=device)
    previous = torch.log_softmax(torch.zeros(VOCAB_SIZE, device=device), dim=-1)

    signals, _ = compute_step_signals(logits, token, previous)
    values = dict(zip(SIGNAL_NAMES, signals.tolist()))

    # dH/dT > 0, so raising the temperature can only raise the entropy.
    curve = [values[n] for n in GRID_SIGNAL_NAMES]
    report.check(
        "H(T) increases monotonically across the whole grid",
        all(a <= b for a, b in zip(curve, curve[1:])),
        " < ".join(f"{v:.3f}" for v in curve),
    )
    report.check("grid point at T=1 equals the reference entropy",
                 abs(values["entropy_at_T1"] - values["entropy"]) < 1e-4)
    report.check("H < ln(V) for a peaked distribution",
                 values["entropy"] < math.log(VOCAB_SIZE))
    report.check("Var_p(z) > 0", values["logit_var"] > 0, f"{values['logit_var']:.4f}")
    report.check("kl_prev finite at t > 0", math.isfinite(values["kl_prev"]),
                 f"{values['kl_prev']:.4f}")
    report.check("signal vector shape and dtype",
                 tuple(signals.shape) == (len(SIGNAL_NAMES),)
                 and signals.dtype == torch.float32)

    # The runtime guard in decoding.py rests on this, so test it in both
    # directions: a clean trajectory must report zero violations, and swapping
    # the band edges must be caught rather than passed through.
    clean = signals.unsqueeze(0)
    report.check("band guard passes a clean step",
                 count_reachable_band_violations(clean) == 0)

    swapped = clean.clone()
    low = SIGNAL_NAMES.index("entropy_at_T0.5")
    high = SIGNAL_NAMES.index("entropy_at_T1.5")
    swapped[:, low], swapped[:, high] = clean[:, high].clone(), clean[:, low].clone()
    report.check("monotonicity guard catches swapped grid points",
                 count_reachable_band_violations(swapped) == 1)


def check_loop_degeneracy(report: Report, device: str) -> None:
    """dH/dT = Var_p(z)/T^3, verified numerically.

    This is the analytic result the conference thesis reports, and the reason the
    sensor reads raw logits rather than tempered ones. Checked by central
    difference in float64 so the finite-difference error stays well below the
    tolerance.
    """
    print("=== 5. dH/dT = Var_p(z)/T^3 (loop-degeneracy result) ===")
    logits = _peaked_logits(device).double()
    token = torch.tensor(42, device=device)
    signals, _ = compute_step_signals(logits.float(), token, None)
    logit_var = dict(zip(SIGNAL_NAMES, signals.tolist()))["logit_var"]

    eps = 1e-4
    numeric = (tempered_entropy(logits, 1.0 + eps)
               - tempered_entropy(logits, 1.0 - eps)) / (2 * eps)
    analytic = entropy_derivative_wrt_temperature(logits.new_tensor(logit_var), 1.0)

    relative_error = abs(numeric.item() - analytic.item()) / abs(numeric.item())
    report.check(
        "central difference matches Var_p(z)", relative_error < 1e-3,
        f"numeric {numeric.item():.6f} vs analytic {analytic.item():.6f},"
        f" rel err {relative_error:.2e}",
    )


def check_precision_floor(report: Report, device: str) -> None:
    """Quantify the bf16 sensor floor, and confirm the fp32 upcast is not cosmetic."""
    print("=== 6. bf16 sensor floor ===")
    logits = _peaked_logits(device)
    exact = tempered_entropy(logits, 1.0).item()
    rounded = tempered_entropy(logits.bfloat16().float(), 1.0).item()
    difference = abs(exact - rounded)

    report.info("H (fp32 logits)", f"{exact:.6f} nats")
    report.info("H (bf16 round trip)", f"{rounded:.6f} nats")
    report.info("sensor floor", f"{difference:.2e} nats"
                                f"  ({difference / exact * 100:.3f}% relative)")
    report.check(
        "bf16 rounding perturbs H, so the fp32 upcast is load-bearing",
        difference > BF16_FLOOR_MIN,
    )


def check_transformers_api(report: Report) -> None:
    """Pin down the API details that must not be guessed."""
    print("=== 7. transformers API ===")
    from transformers import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    for name, model_class in (("Qwen2ForCausalLM", Qwen2ForCausalLM),
                              ("Qwen3ForCausalLM", Qwen3ForCausalLM)):
        parameters = list(inspect.signature(model_class.forward).parameters)
        current = "logits_to_keep" in parameters
        renamed = "num_logits_to_keep" in parameters

        # Exactly one spelling must exist, otherwise the prefill-slicing kwarg
        # would have to be guessed -- and guessing it wrong costs ~1.9 GB at batch 16.
        report.check(f"{name}: exactly one prefill-slicing kwarg", current != renamed,
                     f"logits_to_keep={current}, num_logits_to_keep={renamed}")
        report.check(f"{name}: accepts past_key_values",
                     "past_key_values" in parameters)
        report.check(f"{name}: accepts use_cache", "use_cache" in parameters)

    report.check("DynamicCache importable", DynamicCache is not None)


def main() -> int:
    report = Report()
    check_versions(report)
    device = check_device(report)
    check_sensor_ground_truth(report, device)
    check_reachable_band(report, device)
    check_loop_degeneracy(report, device)
    check_precision_floor(report, device)
    check_transformers_api(report)
    return report.summarise()


if __name__ == "__main__":
    raise SystemExit(main())
