#!/usr/bin/env python
"""Stage 2a: cold HYPOTHESIS arms. Its purpose is `n` on the collapse regime.

Stage 1 returned three repetitive-collapse trajectories out of 400, located by
`find_onset` at steps 1312, 2192 and 5056. All three came from `demo-T0.3`;
`hyp-T0.7`, `hyp-T1.0` and `reference` returned zero between them (0/348).
By the project's own rule, H1a/H1b/H1c may be computed on HYPOTHESIS arms only
-- and those arms contain **no instances of the target regime at all**. That is
not a weak test, it is the absence of one, and no amount of reanalysis of the
existing million steps changes it.

So: the same regime, at the same temperature that produced it, in arms that are
admissible for the hypotheses.

  * Three arms at T in {0.1, 0.2, 0.3}, `top_p = 1.0`, role HYPOTHESIS.
    T = 0.3 reproduces the one operating point where collapse is known to occur
    (3/52 = 5.8%); 0.2 and 0.1 extend the sweep colder, where the greedy-loop
    mechanism predicts a higher rate. Whether it does is itself a result.
  * `n_problems = 124` per arm, identical to `hyp-T0.7` and `hyp-T1.0`. Arms draw
    `problems[:n_problems]` from one ordered list and seed by problem index, so
    124 gives the *same 124 problems under the same seeds* as the existing
    hypothesis arms. Invariant #9 wants temperature comparisons paired within
    problem; this is what makes them paired.
  * `role = HYPOTHESIS` is claimed, not assumed. It requires both classes to
    arise naturally in the arm. HEALTHY certainly will; FAILED_TERMINATION is the
    point of the run. If an arm comes back with an empty class, it demonstrates a
    mechanism and must not be pooled into a hypothesis test -- check before use.

`max_new_tokens` stays 8192, uniform across every arm ever run. One onset sat at
step 5056, so a lower ceiling would have destroyed a third of the evidence, and
a per-arm ceiling would make FAILED_TERMINATION mean different
things in different arms.

124 x 3 = 372 trajectories, a multiple of `BATCH_SIZE` in every arm.

Usage:
    python scripts/run_cold_arms.py [--dry-run] [--resume RUN_DIR]

Long. tmux is not installed on this host; run detached under `setsid nohup`,
logged to file.
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
BATCH_SIZE = 4
CHECKPOINT_EVERY = 20

ARMS = (
    Arm("cold-T0.3", temperature=0.3, top_p=1.0, max_new_tokens=CEILING,
        n_problems=124, role=ArmRole.HYPOTHESIS),
    Arm("cold-T0.2", temperature=0.2, top_p=1.0, max_new_tokens=CEILING,
        n_problems=124, role=ArmRole.HYPOTHESIS),
    Arm("cold-T0.1", temperature=0.1, top_p=1.0, max_new_tokens=CEILING,
        n_problems=124, role=ArmRole.HYPOTHESIS),
)

# Measured on stage 1, not fitted: 400 trajectories in 13 h 45 min at batch 4.
MINUTES_PER_TRAJECTORY_MEASURED = 2.06


def main() -> int:
    total = sum(arm.n_problems for arm in ARMS)
    print(f"model     : {MODEL} @ {REVISION[:12]}")
    print(f"ceiling   : {CEILING} (uniform, as in stage 1)")
    print(f"batch     : {BATCH_SIZE}")
    print(f"checkpoint: every {CHECKPOINT_EVERY} trajectories")
    print(f"trajectories: {total}\n")
    print(f"  {'arm':<12}{'role':<15}{'T':>5}{'top_p':>7}{'n':>6}")
    for arm in ARMS:
        print(f"  {arm.name:<12}{str(arm.role):<15}{arm.temperature:>5}"
              f"{arm.top_p:>7}{arm.n_problems:>6}")

    hours = total * MINUTES_PER_TRAJECTORY_MEASURED / 60
    print(f"\nestimated {hours:.1f} h at the stage-1 measured rate. Cold arms "
          f"truncate\nmore often and truncated trajectories run the full "
          f"{CEILING} steps, so treat\nthat as a floor: 13-17 h is the honest "
          f"interval.")
    print(f"\nA killed run loses at most {CHECKPOINT_EVERY} trajectories; "
          f"resume with\n  python scripts/run_cold_arms.py --resume "
          f"data/runs/<run_dir>")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
