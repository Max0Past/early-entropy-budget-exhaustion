"""Orchestration: model x temperature x problem x seed.

The only module that knows about all the others. Decoding does not know about
storage, storage does not know how trajectories were produced, and data does not
know about either. That separation is what lets the decode loop become the
Phase-II control loop unchanged.

One invocation writes one self-describing run directory. Everything the analysis
will ever need is computed online, because anything not persisted here costs a
full re-run to recover.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cotctl.data import (
    GSM8K_REVISION,
    Problem,
    answers_match,
    extract_pred_answer,
    format_prompt,
    load_gsm8k,
)
from cotctl.decoding import (
    DecodeConfig,
    Trajectory,
    generate_batch,
    generate_trajectory,
)
from cotctl.signals import TEMPERATURE_GRID
from cotctl.storage import (
    ArmRole,
    RunWriter,
    TrajectoryRecord,
    build_steps_table,
    completed_trajectory_ids,
    new_run_id,
)

# Phase I is bf16, never quantised. Quantisation injects asymmetric noise into the
# logit tail: it inflates low-entropy measurements and compresses exactly the
# contrast the sensor exists to detect.
DTYPE = torch.bfloat16


@dataclass(frozen=True, slots=True)
class Arm:
    """One operating point: a (temperature, top_p) PAIR, not a temperature.

    The protocol's arms differ in two variables at once, and an earlier version of
    this config could not express that -- it held a list of temperatures and a
    single global `top_p`, which silently made the healthy reference arm
    unrepresentable. Consequences of getting this wrong are not cosmetic:
    `tau_burst` measured at an artificial `top_p = 1.0` is, by the study protocol, a
    number that means nothing.

    `max_new_tokens` is per arm because it is a ceiling rather than a budget, and
    the arms have genuinely different length distributions. Setting it low on the
    healthy arm truncates healthy trajectories, which by the labelling rule makes
    them FAILED_TERMINATION and leaves `tau_burst` with no substrate at all.

    `n_problems` is per arm because the healthy arm does double duty (tau_burst
    and the corridor baseline) and loses sample twice over, to truncation and to
    wrong answers, while the failure arms only need steps.

    `role` has no default, deliberately. It decides which analyses may include
    the arm, and a default would be silently accepted by every arm that never
    thought about it -- which is exactly how an actuator-limit arm ends up inside
    a base-rate estimate. `name` is the canonical grouping key and must be unique
    within a run; both reach the data as `arm_id` / `arm_role`.
    """

    name: str
    temperature: float
    top_p: float
    max_new_tokens: int
    n_problems: int
    role: ArmRole


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One invocation. Serialised verbatim into the run directory."""

    model_name: str
    model_revision: str
    arms: tuple[Arm, ...]
    dataset: str = "gsm8k"
    dataset_revision: str = GSM8K_REVISION
    split: str = "test"
    n_replicates: int = 1
    # Temperatures at which the sensor evaluates H each step; recorded in
    # config.yaml so a run stays self-describing if the grid ever changes.
    grid: tuple[float, ...] = TEMPERATURE_GRID
    top_k_log: int = 32
    base_seed: int = 20260726
    # Trajectories between durable checkpoints. A Parquet file is unreadable
    # without its footer, so a single streaming file is all-or-nothing and
    # one kill costs the whole run -- 9 to 15 hours for the pilot. 0 keeps
    # the old single-file layout.
    checkpoint_every: int = 0
    # Trajectories generated per forward pass. Sampling uses one generator
    # per row, so this changes throughput and nothing else -- a trajectory
    # must not depend on what it was batched with. check_batching.py is what
    # keeps that true; raise it only once that test passes on this hardware.
    batch_size: int = 1
    # Marks a run directory as evidence rather than data, e.g. "_smoke_".
    # Analysis must skip prefixed directories; they are not deleted, because
    # destroying artefacts in a pipeline whose one claimed virtue is
    # reproducibility is the wrong default -- it has already cost us 3004 steps.
    tag: str = ""

    def __post_init__(self) -> None:
        """Reject configs whose arms cannot be told apart afterwards.

        `arm_id` is the grouping key for every analysis, so duplicate names merge
        two operating points into one bucket silently. Exactly one REFERENCE arm,
        because the corridor and the `tau_burst` baseline are defined from it and
        two candidates means the analysis picks one by accident.
        """
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"arm names must be unique, got {names}")

        references = [a.name for a in self.arms if a.role is ArmRole.REFERENCE]
        if len(references) > 1:
            raise ValueError(
                f"exactly one REFERENCE arm is allowed, got {references}"
            )


def load_model(model_name: str, revision: str, device: str = "cuda"):
    """Load weights and tokenizer at a pinned revision, in bf16.

    The revision is mandatory rather than optional: weights on the Hub are
    updated silently, and an unpinned load makes two runs incomparable without
    raising anything.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=DTYPE
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def trajectory_seed(base_seed: int, problem_index: int, replicate: int) -> int:
    """Seed shared by (problem, replicate) across every temperature arm.

    Common random numbers: the same seed is reused across temperatures so that
    comparisons are paired within problem. It costs nothing and it removes the
    component of the variance that comes from the draw rather than from the arm.
    Temperature is deliberately absent from the derivation.
    """
    return (base_seed + problem_index * 1_000 + replicate) % (2**31 - 1)


def _record_for(
    trajectory: Trajectory,
    traj_id: str,
    problem: Problem,
    config: RunConfig,
    arm: Arm,
    seed: int,
    wall_time_s: float,
) -> TrajectoryRecord:
    """Score one finished trajectory. Scoring only -- no class label.

    Deriving the class belongs to labeling.py, offline, from outcome x
    termination. Doing it here would bake a labelling decision into the data
    collection and make it unrevisable without regenerating.
    """
    extracted = extract_pred_answer(trajectory.text)
    return TrajectoryRecord(
        traj_id=traj_id,
        problem_id=problem.problem_id,
        dataset=config.dataset,
        dataset_revision=config.dataset_revision,
        model_name=config.model_name,
        model_revision=config.model_revision,
        arm_id=arm.name,
        arm_role=str(arm.role),
        batch_size=config.batch_size,
        temperature=arm.temperature,
        top_p=arm.top_p,
        seed=seed,
        n_steps=len(trajectory.token_ids),
        termination=trajectory.termination,
        pred_answer=extracted.value,
        pred_answer_raw=extracted.raw,
        gold_answer=problem.gold_answer,
        extraction_outcome=extracted.outcome,
        correct=answers_match(extracted.value, problem.gold_answer),
        wall_time_s=wall_time_s,
    )


def run(
    config: RunConfig,
    runs_root: Path,
    device: str = "cuda",
    on_trajectory: Callable[[str, int, str], None] | None = None,
    resume: Path | None = None,
) -> Path:
    """Execute one full sweep and write its run directory.

    `on_trajectory(traj_id, n_steps, termination)` is called after each
    trajectory is persisted. Presentation is the caller's business; the runner
    only reports. A multi-hour run with no output is how a stalled loop goes
    unnoticed until the wall clock says so.

    Returns the run directory.

    `resume` points at an existing run directory. Trajectories already durably on
    disk there are skipped and generation continues into new checkpoint parts.
    The run keeps its original `run_id` and manifest, so a resumed run is one run
    rather than two -- which matters because `git_commit` and the model revision
    are part of a trajectory's identity, and a fresh directory would let the same
    experiment be counted twice.

    **Resume is only safe when the config matches.** Nothing here verifies that;
    the caller passes the same config it started with, and the manifest in the
    directory is the record of what that was.
    """
    model, tokenizer = load_model(config.model_name, config.model_revision, device)

    # Load once, for the largest arm, and let each arm take a prefix. Arms then
    # share problems, which keeps comparisons paired within problem -- the same
    # reason the seed does not depend on temperature.
    widest = max(arm.n_problems for arm in config.arms)
    problems = load_gsm8k(config.split, widest, config.dataset_revision)

    if resume is not None:
        run_dir = Path(resume)
        run_id = pq.read_table(
            next(iter(sorted(run_dir.glob("trajectories*.parquet")))),
            columns=["run_id"])["run_id"][0].as_py()
        already = completed_trajectory_ids(run_dir)
        print(f"  [resume] {run_dir.name}: {len(already)} trajectories already "
              f"on disk")
    else:
        run_id = new_run_id()
        run_dir = Path(runs_root) / f"{config.tag}{run_id}"
        already = set()

    env_lock = Path(__file__).resolve().parents[2] / "env.lock.txt"

    with RunWriter(run_dir, asdict(config), env_lock, config.top_k_log,
                   checkpoint_every=config.checkpoint_every,
                   resume=resume is not None) as writer:
        for arm in config.arms:
            decode_config = DecodeConfig(
                temperature=arm.temperature,
                top_p=arm.top_p,
                max_new_tokens=arm.max_new_tokens,
                grid=config.grid,
                top_k_log=config.top_k_log,
            )

            # (problem, replicate, seed) for every trajectory this arm owes,
            # flattened so batching does not have to nest two loops.
            #
            # Skipping happens here rather than inside the batch loop, so a
            # resumed run repacks the remaining work into full batches. Filtering
            # after chunking would leave holes and change the effective batch
            # size, which provably moves the measurement.
            work = [
                (problem_index, problem, replicate,
                 trajectory_seed(config.base_seed, problem_index, replicate))
                for problem_index, problem in enumerate(problems[: arm.n_problems])
                for replicate in range(config.n_replicates)
                if f"{problem.problem_id}-{arm.name}-r{replicate}" not in already
            ]
            if not work:
                continue

            for start in range(0, len(work), config.batch_size):
                chunk = work[start:start + config.batch_size]
                prompts = [format_prompt(p.question, tokenizer) for _, p, _, _ in chunk]
                generators = [
                    torch.Generator(device=device).manual_seed(seed)
                    for _, _, _, seed in chunk
                ]

                started = time.perf_counter()
                if len(chunk) == 1:
                    # The single-stream path stays the reference implementation
                    # that check_batching.py compares against, so batch_size = 1
                    # must not silently route through the batched loop.
                    generated = [generate_trajectory(
                        model, tokenizer, prompts[0], decode_config, generators[0]
                    )]
                else:
                    generated = generate_batch(
                        model, tokenizer, prompts, decode_config, generators
                    )
                elapsed = time.perf_counter() - started
                # Wall time is per batch; attributing all of it to each row would
                # make the column meaningless when batch_size > 1.
                per_trajectory = elapsed / len(chunk)

                for (_, problem, replicate, seed), trajectory in zip(chunk, generated):
                    traj_id = f"{problem.problem_id}-{arm.name}-r{replicate}"
                    writer.write_steps(
                        build_steps_table(
                            run_id,
                            traj_id,
                            trajectory.token_ids,
                            trajectory.signals.numpy(),
                            trajectory.topk_ids.numpy(),
                            trajectory.topk_logits.numpy(),
                            writer.steps_schema,
                        )
                    )
                    writer.write_trajectory(
                        _record_for(
                            trajectory, traj_id, problem, config, arm, seed,
                            per_trajectory,
                        ),
                        run_id,
                    )

                    if on_trajectory is not None:
                        on_trajectory(
                            traj_id, len(trajectory.token_ids), trajectory.termination
                        )

    return run_dir
