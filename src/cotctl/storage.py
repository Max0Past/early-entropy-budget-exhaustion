"""Parquet schema and writers for run output.

One run directory per invocation, self-describing or it is not reproducible:

    data/runs/{run_id}/
        steps.parquet          one row per decoding step
        trajectories.parquet   one row per trajectory
        config.yaml            the arm that produced it
        env.lock.txt           the frozen environment
        git_commit             the code that produced it

Two design points that are not negotiable:

* Schemas are declared explicitly, never inferred from the first batch. An
  inferred schema silently changes type when a column happens to be all-null or
  all-integer in the first trajectory, and the resulting file is unreadable
  against the rest of the run.
* Full logit vectors are never stored. Vocab is ~152k, so a single fp32 vector is
  608 KB per step and the pilot would be ~2 TB. Signals are reduced on-GPU at the
  step; only top-k (k=32) ids and logits survive.

Storage does not know how trajectories were produced. The runner glues decoding,
labeling and storage together; none of them import each other.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cotctl.signals import (
    N_BASE_SIGNALS,
    TEMPERATURE_GRID,
    T_MAX_INDEX,
    T_MIN_INDEX,
)

TOP_K: Final[int] = 32


class ArmRole(StrEnum):
    """What an arm is *for*. Determines which analyses may include it.

    Temperature does not carry this. `temperature` says an arm ran at 1.3; it
    does not say that pooling that arm into a base-rate estimate is invalid, and
    the person doing the pooling three months from now will not know either.

    Measured consequence rather than a hypothetical one: the H1a AUC across all
    arms is 0.854, and stratified within arm it is 0.600 — the cross-arm figure
    is mostly the temperature, because the positive class is 5/9 `DEMONSTRATION`
    and the negative class is mostly `REFERENCE`. Role has to be in the schema
    for a stratified comparison to be expressible at all.

    The split between `HYPOTHESIS` and `DEMONSTRATION` follows from that. If
    every H1a comparison is stratified by arm, testing it needs **both classes
    inside one arm**. At T = 1.3 everything truncates, so `HEALTHY` is empty and
    H1a is untestable there by construction; T = 0.3 fails the same way from the
    other side. An arm that cannot supply both classes can still demonstrate a
    mechanism, but it cannot test a hypothesis, and the two must not be pooled.
    """

    # The healthy reference arm. Sole source of the corridor and of the
    # `tau_burst` baseline, which are computed once here and applied unchanged to
    # every other arm.
    REFERENCE = "reference"
    # Both classes arise naturally here, so H1a and H1c are computed on these
    # arms and only on these arms. T in {0.7, 1.0}.
    HYPOTHESIS = "hypothesis"
    # Demonstrates the limit of the actuator and the irreversibility mechanism.
    # Driven deliberately past the admissible band, so its class counts are
    # manufactured: never pooled into a base rate, never an AUC. T in {0.3, 1.3}.
    DEMONSTRATION = "demonstration"


#: Roles on which a hypothesis may be tested. Anything else is descriptive.
TESTABLE_ROLES: Final[frozenset[str]] = frozenset({ArmRole.HYPOTHESIS})


class _ConfigDumper(yaml.SafeDumper):
    """SafeDumper that can write the enums a RunConfig contains.

    `yaml.safe_dump` dispatches on the exact type, so a `StrEnum` misses the
    `str` representer and raises even though it *is* a str. The config write is
    the first thing a run does, so without this every run dies before generating
    a token -- which is what happened the first time an arm carried a role.
    """


_ConfigDumper.add_multi_representer(
    StrEnum, lambda dumper, data: dumper.represent_str(str(data))
)

# Row groups smaller than this are inefficient to read back; larger ones cost
# memory while buffering. Trajectories are flushed in whole, never split.
ROW_GROUP_TARGET: Final[int] = 50_000

COMPRESSION: Final[str] = "zstd"


def steps_schema(top_k: int = TOP_K, n_grid: int = len(TEMPERATURE_GRID)) -> pa.Schema:
    """One row per decoding step.

    `kl_prev` is null at t = 0 by construction. `entropy_at_t_min` /
    `entropy_at_t_max` are the two edges of the reachable entropy band and exist
    so the controllability map can be computed analytically, without re-running.
    """
    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("traj_id", pa.string(), nullable=False),
            pa.field("step", pa.int32(), nullable=False),
            pa.field("token_id", pa.int32(), nullable=False),
            pa.field("entropy", pa.float32(), nullable=False),
            pa.field("margin", pa.float32(), nullable=False),
            pa.field("surprisal", pa.float32(), nullable=False),
            pa.field("kl_prev", pa.float32(), nullable=True),
            pa.field("logit_var", pa.float32(), nullable=False),
            # Edges of the admissible actuator band. Also present inside
            # entropy_grid, but kept named because they mean something the other
            # grid points do not, and because it keeps earlier runs comparable.
            pa.field("entropy_at_t_min", pa.float32(), nullable=False),
            pa.field("entropy_at_t_max", pa.float32(), nullable=False),
            # H(T) across the whole grid, so the temperature that would reach a
            # target can be read off by interpolation instead of linearised.
            pa.field("entropy_grid", pa.list_(pa.float32(), n_grid), nullable=False),
            pa.field("topk_ids", pa.list_(pa.int32(), top_k), nullable=False),
            pa.field("topk_logits", pa.list_(pa.float32(), top_k), nullable=False),
        ]
    )


def trajectories_schema() -> pa.Schema:
    """One row per trajectory.

    `model_revision` and `dataset_revision` are the pinned commit SHAs. Both are
    nullable only so that a smoke run against a moving ref can still be written;
    a real run without them is not reproducible and analysis should reject it.

    `pred_answer` is null when extraction failed, which is deliberately distinct
    from an extracted-but-wrong answer: an extractor failure otherwise
    masquerades as SEMANTIC_ERROR, the negative control for H1c.
    """
    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("traj_id", pa.string(), nullable=False),
            pa.field("problem_id", pa.string(), nullable=False),
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("dataset_revision", pa.string(), nullable=True),
            pa.field("model_name", pa.string(), nullable=False),
            pa.field("model_revision", pa.string(), nullable=True),
            # The canonical grouping key. `temperature` is float32 and cannot be
            # one: 0.6 round-trips as 0.6000000238418579, so grouping on it
            # depends on how the reader coerces, and two arms that differ in
            # `top_p` collapse into one bucket. Group on `arm_id`.
            pa.field("arm_id", pa.string(), nullable=False),
            pa.field("arm_role", pa.string(), nullable=False),
            # Batching provably moves the measurement: batched GEMMs reduce in a
            # different order, so logits shift by multiples of the bf16 ULP
            # (0.125 at a top-1 near 25.5) and |dH| reaches a few times 1e-2.
            # Not a bug -- see scripts/diagnose_batch_precision.py -- but it
            # means two runs at different batch sizes are comparable only to that
            # precision, and the corridor must come from the same batch size as
            # the data it is applied to.
            pa.field("batch_size", pa.int32(), nullable=False),
            pa.field("temperature", pa.float32(), nullable=False),
            pa.field("top_p", pa.float32(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("git_commit", pa.string(), nullable=True),
            pa.field("n_steps", pa.int32(), nullable=False),
            pa.field("termination", pa.string(), nullable=False),
            pa.field("pred_answer", pa.string(), nullable=True),
            # The matched span before normalisation. The pipeline requires a
            # hand-audit of ~50 extractions; without the raw span that audit means
            # regenerating the run instead of reading this file.
            pa.field("pred_answer_raw", pa.string(), nullable=True),
            pa.field("gold_answer", pa.string(), nullable=False),
            pa.field("extraction_outcome", pa.string(), nullable=False),
            pa.field("correct", pa.bool_(), nullable=False),
            pa.field("wall_time_s", pa.float32(), nullable=False),
        ]
    )


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """The trajectory-level row. Mirrors `trajectories_schema` field for field."""

    traj_id: str
    problem_id: str
    dataset: str
    dataset_revision: str | None
    model_name: str
    model_revision: str | None
    arm_id: str
    arm_role: str
    batch_size: int
    temperature: float
    top_p: float
    seed: int
    n_steps: int
    termination: str
    pred_answer: str | None
    pred_answer_raw: str | None
    gold_answer: str
    extraction_outcome: str
    correct: bool
    wall_time_s: float


def current_git_commit(repo_root: Path | None = None) -> str | None:
    """Resolve HEAD, or None outside a repository.

    Part of the experimental record: without it a run directory cannot be tied
    back to the code that produced it.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def new_run_id() -> str:
    """UTC timestamp plus a short random suffix, sortable and collision-free."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _fixed_size_list(values: np.ndarray, dtype: pa.DataType) -> pa.Array:
    """Build a FixedSizeListArray from a 2-D (n_rows, list_size) array."""
    n_rows, list_size = values.shape
    flat = pa.array(values.reshape(-1), type=dtype)
    return pa.FixedSizeListArray.from_arrays(flat, list_size)


def build_steps_table(
    run_id: str,
    traj_id: str,
    token_ids: list[int],
    signals: np.ndarray,
    topk_ids: np.ndarray,
    topk_logits: np.ndarray,
    schema: pa.Schema,
) -> pa.Table:
    """Assemble one trajectory's step rows.

    Args:
        signals: (n_steps, 7), column order must match `cotctl.signals.SIGNAL_NAMES`.
        topk_ids: (n_steps, top_k) int32.
        topk_logits: (n_steps, top_k) float32.
    """
    n_steps = len(token_ids)
    if signals.shape[0] != n_steps:
        raise ValueError(
            f"signals has {signals.shape[0]} rows but {n_steps} tokens were generated"
        )

    columns = {
        "run_id": pa.array([run_id] * n_steps, type=pa.string()),
        "traj_id": pa.array([traj_id] * n_steps, type=pa.string()),
        "step": pa.array(np.arange(n_steps, dtype=np.int32)),
        "token_id": pa.array(np.asarray(token_ids, dtype=np.int32)),
        "entropy": pa.array(signals[:, 0].astype(np.float32)),
        "margin": pa.array(signals[:, 1].astype(np.float32)),
        "surprisal": pa.array(signals[:, 2].astype(np.float32)),
        # NaN at t = 0 becomes a real null, so downstream aggregation skips it
        # instead of silently poisoning a mean.
        "kl_prev": pa.array(
            signals[:, 3].astype(np.float32), mask=np.isnan(signals[:, 3])
        ),
        "logit_var": pa.array(signals[:, 4].astype(np.float32)),
        "entropy_at_t_min": pa.array(
            signals[:, N_BASE_SIGNALS + T_MIN_INDEX].astype(np.float32)),
        "entropy_at_t_max": pa.array(
            signals[:, N_BASE_SIGNALS + T_MAX_INDEX].astype(np.float32)),
        "entropy_grid": _fixed_size_list(
            signals[:, N_BASE_SIGNALS:].astype(np.float32), pa.float32()),
        "topk_ids": _fixed_size_list(topk_ids.astype(np.int32), pa.int32()),
        "topk_logits": _fixed_size_list(topk_logits.astype(np.float32), pa.float32()),
    }
    return pa.Table.from_pydict(columns, schema=schema)


def read_steps(run_dir: Path, columns: list[str] | None = None) -> pa.Table:
    """Steps of one run, whether written as one file or as checkpoint parts.

    `columns` projects at read time. The two top-k columns are 32-wide lists
    and dominate the file; skipping them turns a 1M-step run from a
    two-minute read into a few seconds.
    """
    return _read_shards(Path(run_dir), "steps", columns)


def read_trajectories(run_dir: Path) -> pa.Table:
    """Trajectories of one run, whether written as one file or as parts."""
    return _read_shards(Path(run_dir), "trajectories")


def _read_shards(
    run_dir: Path, stem: str, columns: list[str] | None = None
) -> pa.Table:
    """Concatenate `stem.parquet` and any `stem-NNNN.parquet`, in order.

    Runs written before checkpointing existed have the single file; runs written
    with it have numbered parts. Both are read the same way so no analysis needs
    to know which it is looking at.

    A shard without a Parquet footer is **skipped with a warning**, not raised on.
    That is the normal state of the part a live run is currently writing, and an
    analysis must be able to read a pilot's completed checkpoints while the pilot
    is still going. The warning is loud because the same condition, on a run that
    is *not* live, means data was lost.
    """
    shards = []
    single = run_dir / f"{stem}.parquet"
    if single.exists():
        shards.append(single)
    shards.extend(sorted(run_dir.glob(f"{stem}-[0-9]*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no {stem} parquet under {run_dir}")

    tables = []
    for shard in shards:
        try:
            tables.append(pq.read_table(shard, columns=columns))
        except (pa.ArrowInvalid, OSError):
            print(f"  [skip] {run_dir.name}/{shard.name}: no Parquet footer "
                  "(still being written, or the writer died)")
    if not tables:
        raise FileNotFoundError(
            f"no readable {stem} parquet under {run_dir}: every shard lacks a "
            "footer")
    return pa.concat_tables(tables)


def completed_trajectory_ids(run_dir: Path) -> set[str]:
    """`traj_id`s already durably on disk in `run_dir`.

    Only fully written parts count. A part whose footer never got written is
    unreadable and is skipped with a warning rather than aborting the resume:
    the trajectories inside it will simply be regenerated, which is the correct
    outcome for a process that died mid-write.
    """
    run_dir = Path(run_dir)
    done: set[str] = set()
    candidates = [run_dir / "trajectories.parquet"]
    candidates += sorted(run_dir.glob("trajectories-[0-9]*.parquet"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            done.update(pq.read_table(path, columns=["traj_id"])["traj_id"].to_pylist())
        except (pa.ArrowInvalid, OSError) as exc:
            print(f"  [resume] {path.name} is unreadable ({exc}); "
                  "its trajectories will be regenerated")
    return done


class RunWriter:
    """Streaming writer for one run directory.

    Steps are buffered and flushed in whole trajectories, so a row group never
    splits a trajectory. Use as a context manager; `close` finalises the Parquet
    footers, without which the files are unreadable.

    **Checkpointing.** With `checkpoint_every` set, the writers are closed and
    reopened on new numbered parts every N trajectories, so everything before the
    last checkpoint is complete and readable even if the process is killed. A
    Parquet file needs its footer to be readable at all, so a single streaming
    file is all-or-nothing -- which for a 9-to-15-hour pilot means one kill costs
    the entire run.

    Parts are `steps-0000.parquet` / `trajectories-0000.parquet`. `read_steps` and
    `read_trajectories` hide the difference from every reader.
    """

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        env_lock: Path | None = None,
        top_k: int = TOP_K,
        checkpoint_every: int = 0,
        resume: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=resume)

        self.steps_schema = steps_schema(top_k)
        self.trajectories_schema = trajectories_schema()
        self.checkpoint_every = checkpoint_every

        self._buffer: list[pa.Table] = []
        self._buffered_rows = 0
        self._since_checkpoint = 0
        # Resuming must not overwrite existing parts, so numbering continues past
        # whatever is already there.
        self._part = (1 + max(
            (int(p.stem.rsplit("-", 1)[1])
             for p in self.run_dir.glob("steps-[0-9]*.parquet")),
            default=-1)) if resume else 0

        # Set before any write, not inside the manifest: a resumed run skips the
        # manifest (it already exists and must not be overwritten with a
        # different commit) but still stamps every row it writes.
        self.git_commit = current_git_commit()
        self._open_writers()

        if not resume:
            self._write_manifest(config, env_lock)

    def _paths(self) -> tuple[Path, Path]:
        if not self.checkpoint_every:
            return (self.run_dir / "steps.parquet",
                    self.run_dir / "trajectories.parquet")
        return (self.run_dir / f"steps-{self._part:04d}.parquet",
                self.run_dir / f"trajectories-{self._part:04d}.parquet")

    def _open_writers(self) -> None:
        steps_path, traj_path = self._paths()
        self._steps_writer = pq.ParquetWriter(
            steps_path, self.steps_schema, compression=COMPRESSION)
        self._traj_writer = pq.ParquetWriter(
            traj_path, self.trajectories_schema, compression=COMPRESSION)

    def checkpoint(self) -> None:
        """Finalise the current parts and start new ones. Safe to call anytime."""
        self._flush_steps()
        self._steps_writer.close()
        self._traj_writer.close()
        self._part += 1
        self._since_checkpoint = 0
        self._open_writers()

    def _write_manifest(self, config: dict[str, Any], env_lock: Path | None) -> None:
        """Everything needed to tie this run to its code and environment."""
        # `self.git_commit` is set in __init__ because a resumed run needs it
        # without rewriting the manifest. It is also a per-row column, because
        # until recently it was declared, documented, and null in every row.
        (self.run_dir / "git_commit").write_text(
            f"{self.git_commit or 'unknown'}\n")
        (self.run_dir / "config.yaml").write_text(
            yaml.dump(config, Dumper=_ConfigDumper, sort_keys=False,
                      allow_unicode=True)
        )
        if env_lock is not None and Path(env_lock).exists():
            shutil.copy2(env_lock, self.run_dir / "env.lock.txt")

    def write_steps(self, table: pa.Table) -> None:
        self._buffer.append(table)
        self._buffered_rows += table.num_rows
        if self._buffered_rows >= ROW_GROUP_TARGET:
            self._flush_steps()

    def _flush_steps(self) -> None:
        if not self._buffer:
            return
        self._steps_writer.write_table(pa.concat_tables(self._buffer))
        self._buffer.clear()
        self._buffered_rows = 0

    def write_trajectory(self, record: TrajectoryRecord, run_id: str) -> None:
        # asdict, not __dict__: TrajectoryRecord uses slots and has no __dict__.
        # `git_commit` comes from the writer rather than the record: the runner
        # should not have to know how the code is versioned, and a per-row copy
        # is what makes (arm, problem, seed, commit, revisions) a usable
        # identity when several runs are pooled.
        row = {"run_id": run_id, "git_commit": self.git_commit, **asdict(record)}
        self._traj_writer.write_table(
            pa.Table.from_pylist([row], schema=self.trajectories_schema)
        )

        # Steps are flushed here rather than at ROW_GROUP_TARGET when
        # checkpointing: a checkpoint is only durable if the steps belonging to
        # the trajectories in it are in the same part.
        self._since_checkpoint += 1
        if self.checkpoint_every and self._since_checkpoint >= self.checkpoint_every:
            self.checkpoint()

    def close(self) -> None:
        self._flush_steps()
        self._steps_writer.close()
        self._traj_writer.close()

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
