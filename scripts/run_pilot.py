#!/usr/bin/env python
"""Stage 1 of the pilot. Its purpose is the base RATE, not the result.

Sized by a power analysis on H1c, because H1c is the kill
criterion. The full pilot needs 115 trajectories per hypothesis arm — but that
number is a function of `p(SEMANTIC_ERROR)`, whose 95% interval is currently
[3%, 56%], so the per-arm requirement ranges 42 to 913. **A factor of 22 is not a
plan.** Stage 1 measures the rate to ±5%; stage 2 is sized from what it returns.

Composition, and why each arm is here:

  * two `HYPOTHESIS` arms, 124 each. Both classes arise naturally at T ∈ {0.7,
    1.0}, so these are the only arms H1a and H1c may be computed on, and the
    stratified estimator needs both classes inside each one.
  * the `REFERENCE` arm, 100. The corridor and the `tau_burst` baseline come from
    here and nowhere else, and it currently rests on 8 trajectories.
  * the `DEMONSTRATION` arm at T = 0.3, 52. This is the arm where slow collapse
    was found, and all 7 observed non-returning excursions came from T ∈ {1.0,
    1.3} — none from here. `tau_trigger` for the collapse regime cannot be set
    until that changes.

T = 1.3 is deliberately **not** in stage 1. Its role is to demonstrate the
actuator limit, it already did so at n = 5 with 100% truncation, and rerunning it
at the new ceiling costs up to 10.5 min per trajectory to re-demonstrate a known
result. It belongs to stage 2.

**One ceiling for every arm**: `max_new_tokens` is the
detector for `FAILED_TERMINATION`, so per-arm ceilings would make that label mean
different things in different arms and H1a would compare labels produced under
different rules. 8192 is provisional (p99.5 of the fitted healthy length law,
interval [2996, 18551]) and stage 1 re-estimates it.

**One replicate per problem.** The Hanley-McNeil sizing assumes independent
trajectories; several seeds on one problem are correlated and invariant #9's
cluster bootstrap would apply, inflating n by the design effect.

Usage:
    python scripts/run_pilot.py [--dry-run] [--resume RUN_DIR]

Long. Run under tmux, logged to file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cotctl.runner import Arm, ArmRole, RunConfig, run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "data" / "runs"

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

CEILING = 8192
BATCH_SIZE = 4          # raise only against check_batching.py on this hardware

# Trajectories between durable checkpoints. A Parquet file is unreadable
# without its footer, so without this one kill costs the whole run -- 9 to 15
# hours. At batch 4 a checkpoint every 20 trajectories is roughly every half
# hour, which bounds what a crash can destroy at that, and lands on a batch
# boundary so no batch is split across parts.
CHECKPOINT_EVERY = 20

# Every arm size is a multiple of BATCH_SIZE, deliberately. A trailing chunk
# smaller than the batch runs at a different effective batch size, and batch
# size provably moves the measurement by multiples of the bf16 ULP -- so a
# short last chunk would put a handful of trajectories on a slightly different
# sensor than the rest of the same arm.

ARMS = (
    Arm("reference", temperature=0.6, top_p=0.95, max_new_tokens=CEILING,
        n_problems=100, role=ArmRole.REFERENCE),
    Arm("hyp-T0.7", temperature=0.7, top_p=1.0, max_new_tokens=CEILING,
        n_problems=124, role=ArmRole.HYPOTHESIS),
    Arm("hyp-T1.0", temperature=1.0, top_p=1.0, max_new_tokens=CEILING,
        n_problems=124, role=ArmRole.HYPOTHESIS),
    Arm("demo-T0.3", temperature=0.3, top_p=1.0, max_new_tokens=CEILING,
        n_problems=52, role=ArmRole.DEMONSTRATION),
)

# From the fitted healthy length law, capped at the ceiling, at 13.0 tok/s.
MINUTES_PER_TRAJECTORY_BATCH1 = 2.6


def main() -> int:
    total = sum(arm.n_problems for arm in ARMS)
    print(f"model     : {MODEL} @ {REVISION[:12]}")
    print(f"ceiling   : {CEILING} (uniform across arms)")
    print(f"batch     : {BATCH_SIZE}")
    print(f"checkpoint: every {CHECKPOINT_EVERY} trajectories")
    print(f"trajectories: {total}\n")
    print(f"  {'arm':<12}{'role':<15}{'T':>5}{'top_p':>7}{'n':>6}")
    for arm in ARMS:
        print(f"  {arm.name:<12}{str(arm.role):<15}{arm.temperature:>5}"
              f"{arm.top_p:>7}{arm.n_problems:>6}")

    hours = total * MINUTES_PER_TRAJECTORY_BATCH1 / 60
    print(f"\nestimated {hours:.0f} h at batch 1; batching is what makes this "
          f"affordable.")
    print(f"Run detached, logged to file. A killed run loses at most the\n"
          f"{CHECKPOINT_EVERY} trajectories since the last checkpoint; resume with\n"
          f"  python scripts/run_pilot.py --resume data/runs/<run_dir>")

    if "--dry-run" in sys.argv:
        print("\n--dry-run: nothing generated.")
        return 0

    resume = None
    if "--resume" in sys.argv:
        resume = Path(sys.argv[sys.argv.index("--resume") + 1])
        print(f"\nresuming {resume.name}")

    config = RunConfig(
        model_name=MODEL,
        model_revision=REVISION,
        arms=ARMS,
        n_replicates=1,
        batch_size=BATCH_SIZE,
        checkpoint_every=CHECKPOINT_EVERY,
    )

    done = 0

    def report(traj_id: str, n_steps: int, termination: str) -> None:
        nonlocal done
        done += 1
        print(f"  [{done:>4}/{total}] {traj_id:<32} {n_steps:>5} {termination}",
              flush=True)

    run_dir = run(config, RUNS_ROOT, on_trajectory=report, resume=resume)
    print(f"\nwrote {run_dir}")
    print("\nNext: scripts/run_cold_arms.py, then reproduce.sh on both runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
