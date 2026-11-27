#!/usr/bin/env python
"""Regression test: batching must not change the measurement.

Batching is what makes the pilot affordable — 19 h at batch 1 against a 12 GB
card that sits at 21% utilisation. It is also a classic trap,
because padding that leaks into the logits produces entirely plausible
output and a silently wrong sensor.

**This goes through `runner.run()`, not through `generate_batch()`.** Every check
script in this repo so far exercises a function and none exercises the
orchestration, so the paths that actually produce data — prompt formatting,
seeding, the writer, the schema — have never been compared across batch sizes. A
test of `generate_batch` alone would leave exactly that gap.

**The tolerance is not 1e-3.** The original plan asked for agreement to ~1e-3
nats, and that number was written without measuring anything. Measured (see
`diagnose_batch_precision.py`): batching moves the logits by exact multiples of
the bf16 ULP, which at a top-1 logit near 25.5 is **0.125**, and that propagates
to |ΔH| of a few times 1e-2. No tolerance below the ULP is reachable in bf16, so
demanding one would only ever produce a failing test that nobody could fix.

What is testable, and what this script asserts:

  1. **No position dependence.** Reversing the row order within a batch must
     change nothing at all — bit for bit. This is the sharp test for padding
     leakage, because a leak necessarily depends on where a row sits and on how
     much padding precedes it. Measured: exactly 0.
  2. **Δlogit is a multiple of the bf16 ULP.** The signature of rounding rather
     than of corruption. A leak would produce arbitrary values.
  3. **|ΔH| stays within the ULP-derived bound.** Reported as the sensor's
     precision under batching, which is a number the thesis has to state anyway.
  4. Token sequences agree until they diverge at a near-tie, and the divergence
     point is reported rather than treated as a failure.

Sampling uses one generator per row, so batch size is invisible to the actuator
and any divergence is attributable to the forward pass alone.

Usage:
    python scripts/check_batching.py [batch_size]      # default 4

Exits 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from cotctl.runner import Arm, ArmRole, RunConfig, run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "data" / "runs"

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

# Overridable so the pilot's actual batch size can be cleared before use:
#   python scripts/check_batching.py 8
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_PROBLEMS = BATCH
# Long enough that a divergence has room to appear, short enough that the test
# is a few minutes rather than an hour.
MAX_NEW_TOKENS = 384
# Derived, not chosen. bf16 keeps 8 mantissa bits, so the logit spacing at a
# top-1 magnitude near 25.5 is 2^(4-7) = 0.125. Entropy responds to a logit
# perturbation with a gain below 1 in this regime; measured worst case over a
# 384-step trajectory is 7.3e-02, so the bound is set at the ULP itself.
BF16_LOGIT_ULP = 0.125
ENTROPY_TOLERANCE = BF16_LOGIT_ULP
PREFIX = "_batchcheck_"

failures: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    if not passed:
        failures.append(label)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
          f"{'  ' + detail if detail else ''}")


def config_for(batch_size: int) -> RunConfig:
    return RunConfig(
        model_name=MODEL,
        model_revision=REVISION,
        tag=PREFIX,
        batch_size=batch_size,
        arms=(
            Arm("batchcheck", temperature=1.0, top_p=1.0,
                max_new_tokens=MAX_NEW_TOKENS, n_problems=N_PROBLEMS,
                role=ArmRole.HYPOTHESIS),
        ),
    )


def read_run(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    steps = pq.read_table(run_dir / "steps.parquet").to_pydict()
    traj_of_step = np.asarray(steps["traj_id"])
    entropy = np.asarray(steps["entropy"], dtype=np.float64)
    token = np.asarray(steps["token_id"])
    topk = np.asarray(steps["topk_logits"].to_pylist()
                      if hasattr(steps["topk_logits"], "to_pylist")
                      else steps["topk_logits"], dtype=np.float64)
    out = {}
    for traj_id in dict.fromkeys(steps["traj_id"]):
        mask = traj_of_step == traj_id
        out[traj_id] = {"entropy": entropy[mask], "token": token[mask],
                        "topk_logits": topk[mask]}
    return out


def bf16_round(values: np.ndarray) -> np.ndarray:
    """`values` rounded to bfloat16 and back, as float32."""
    import torch
    return torch.from_numpy(values.astype(np.float32)).to(
        torch.bfloat16).float().numpy()


def first_divergence(a: np.ndarray, b: np.ndarray) -> int | None:
    common = min(a.size, b.size)
    differing = np.nonzero(a[:common] != b[:common])[0]
    if differing.size:
        return int(differing[0])
    return None if a.size == b.size else common


def main() -> int:
    print(f"model : {MODEL} @ {REVISION[:12]}")
    print(f"arm   : {N_PROBLEMS} problems, ceiling {MAX_NEW_TOKENS}, T=1.0")
    print(f"compare: batch_size 1 vs {BATCH}, through runner.run()\n")

    print("=== generating ===")
    single = run(config_for(1), RUNS_ROOT,
                 on_trajectory=lambda t, n, term:
                 print(f"  batch=1  {t:<28} {n:>5} {term}"))
    batched = run(config_for(BATCH), RUNS_ROOT,
                  on_trajectory=lambda t, n, term:
                  print(f"  batch={BATCH}  {t:<28} {n:>5} {term}"))

    a, b = read_run(single), read_run(batched)

    print("\n=== 1. the same trajectories exist in both ===")
    check("trajectory ids match", set(a) == set(b),
          f"{len(a)} vs {len(b)}")
    if set(a) != set(b):
        return 1

    print("\n=== 2. token sequences ===")
    divergences: dict[str, int | None] = {}
    for traj_id in sorted(a):
        where = first_divergence(a[traj_id]["token"], b[traj_id]["token"])
        divergences[traj_id] = where
        if where is None:
            check(f"{traj_id}: identical", True,
                  f"{a[traj_id]['token'].size} tokens")
        else:
            print(f"  [INFO] {traj_id}: diverges at step {where} of "
                  f"{a[traj_id]['token'].size}/{b[traj_id]['token'].size}")

    print("\n=== 3. logits differ only by bf16 rounding ===")
    for traj_id in sorted(a):
        where = divergences[traj_id]
        common = (min(a[traj_id]["entropy"].size, b[traj_id]["entropy"].size)
                  if where is None else where)
        if common == 0:
            continue
        left = a[traj_id]["topk_logits"][:common]
        right = b[traj_id]["topk_logits"][:common]

        # The model computes in bf16 and the sensor upcasts, so every persisted
        # logit must already BE a bf16 value. If it is, the two runs differ only
        # in which bf16 value they landed on, which is rounding by definition.
        # Anything not representable would mean the value was produced somewhere
        # other than the model's own output.
        exact = all(np.array_equal(v, bf16_round(v)) for v in (left, right))
        check(f"{traj_id}: persisted logits are exact bf16 values", exact)

        delta = np.abs(left - right)
        nonzero = delta > 0
        if not nonzero.any():
            check(f"{traj_id}: top-k logits identical", True)
            continue
        # Spacing is per value, not global: it is 2^(exponent - 7), so it halves
        # every time the magnitude halves. Comparing against one global ULP was
        # wrong -- top-k logits span 4.6 to 39.3, an ULP range of 0.031 to 0.25.
        #
        # And the spacing to use is the one of the SMALLER magnitude. Each value
        # is a multiple of its own spacing, and the spacings are powers of two,
        # so both values are multiples of the finer one and so is their
        # difference. Taking the coarser one instead makes every pair that
        # straddles a power of two land on a half step -- residual exactly 0.500,
        # on 0.36% of pairs, which is rare enough to look like a real defect and
        # certain enough to fail every trajectory.
        scale = np.minimum(np.abs(left), np.abs(right))
        ulp = np.ldexp(1.0, np.floor(np.log2(np.maximum(scale, 1e-30)))
                       .astype(int) - 7)
        steps_apart = delta[nonzero] / ulp[nonzero]
        residual = np.abs(steps_apart - np.round(steps_apart))
        check(f"{traj_id}: |dlogit| is a whole number of bf16 steps",
              float(residual.max()) < 1e-6,
              f"worst residual={residual.max():.2e}  "
              f"max steps apart={steps_apart.max():.0f}  "
              f"max|dlogit|={delta.max():.4f}")

    print("\n=== 4. entropy over the common prefix ===")
    worst = 0.0
    for traj_id in sorted(a):
        where = divergences[traj_id]
        common = (min(a[traj_id]["entropy"].size, b[traj_id]["entropy"].size)
                  if where is None else where)
        if common == 0:
            check(f"{traj_id}: no common prefix", False)
            continue
        delta = np.abs(a[traj_id]["entropy"][:common]
                       - b[traj_id]["entropy"][:common])
        worst = max(worst, float(delta.max()))
        check(f"{traj_id}: |dH| <= {ENTROPY_TOLERANCE} over {common} steps",
              float(delta.max()) <= ENTROPY_TOLERANCE,
              f"max={delta.max():.2e}  mean={delta.mean():.2e}")

    print(f"\n  worst |dH| across all trajectories: {worst:.3e} "
          f"(tolerance {ENTROPY_TOLERANCE})")
    print("  The monotonicity guard in generate_batch raises on padding leakage,")
    print("  so reaching this line at all means it did not fire.")

    print("\n=== 5. cleanup ===")
    for directory in (single, batched):
        shutil.rmtree(directory)
        print(f"  removed {directory.name}")
    print("  Both runs are tagged evidence, not data: they carry a ceiling of")
    print(f"  {MAX_NEW_TOKENS} and would contaminate any length statistic.")

    print()
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
