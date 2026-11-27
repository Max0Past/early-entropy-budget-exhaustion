"""Loading runs off disk, with identity enforced rather than assumed.

Analysis pools run directories. Pooling is where a whole class of silent errors
lives, and one of them already happened: the healthy reference arm is
deterministic and was regenerated alongside a later failure arm, so pooling every
directory counted 11 of 41 trajectories twice — 36% of the healthy step pool the
corridor and the `tau_burst` baseline were computed on. Nothing raised.

The fix is not deduplication. Deduplicating on everything in sight would also
silently collapse two trajectories that are *supposed* to differ, because they
were produced by different code or a different model revision, and picking one of
those by timestamp is a decision no analysis should make quietly.

So identity is two-level:

* **Experimental identity** is `(problem, temperature, top_p, seed)`. Two records
  sharing it but differing in code or revision are a regeneration; the newer one
  supersedes, and what was superseded is returned so it can be printed.
* **Full identity** adds `(git_commit, model_revision, dataset_revision)`. Two
  records sharing *that* are the same experiment run twice, which is a bug in how
  the runs were assembled, and `load_runs` **raises**.

**`arm` is deliberately not in either key.** It was, and that is how 77
trajectories came to be counted twice. Measured over every run directory: the old
key caught 11 collisions, this one catches 88, and the 77 it had been missing sit
across three renames of the same configuration --

    cold-T0.3 / demo-T0.3      52     (5 of them also fail-T0.3)
    healthy   / reference      10
    fail-T0.7 / hyp-T0.7        5
    fail-T1.0 / hyp-T1.0        5

-- because the guard compared two different strings and saw two different
experiments. An arm name is free text a human types into a run script; it cannot
be load-bearing, and it was renamed three times without anyone intending to
defeat anything. What actually decides whether two trajectories are the same
experiment is the problem, the sampler settings and the seed, and all four are in
the schema.

`temperature` and `top_p` are float32 on disk, where `0.6` round-trips as
`0.6000000238418579` and the exact value therefore depends on how the reader
coerces -- which is why the schema comment says not to
group on them. `_grid` removes that objection by quantising to 1e-3 before
comparison, so the key is an integer and reader-independent. The grid is far
coarser than the float32 error (~2e-8) and far finer than any temperature this
study distinguishes.

What the key deliberately does *not* carry is the ceiling, which is a property
of the run rather than of the trajectory. A consequence worth stating: the 2048-
and 4096-token diagnostics now collapse into the 8192 pilot instead of both
surviving, because they are the same problem at the same settings. That is safer
than counting them twice, but it is not a licence to pool them -- labels depend
on the ceiling, and `only=` is still the way to keep runs apart.

**Role conflicts are resolved conservatively.** If two records share an
experimental identity but declare different `arm_role`, the survivor keeps the
*less* permissive role. `demo-T0.3` was declared `demonstration` and `cold-T0.3`
`hypothesis`, so the 52 collisions stay non-testable rather than being quietly
promoted into the pool a hypothesis may be tested on. Superseding by timestamp
alone would have moved them the other way, and `require_testable` would not have
noticed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from cotctl.labeling import TrajectoryClass, classify
from cotctl.storage import (
    TESTABLE_ROLES,
    ArmRole,
    read_steps,
    read_trajectories,
)

#: Roles for the three diagnostic runs written before `arm_role` existed. This
#: mapping exists so the inference is visible and auditable instead of implicit;
#: it is not a fallback for new runs, which carry the column.
LEGACY_ARM_ROLES: Final[dict[str, ArmRole]] = {
    "healthy": ArmRole.REFERENCE,
    "fail-T0.7": ArmRole.HYPOTHESIS,
    "fail-T1.0": ArmRole.HYPOTHESIS,
    "fail-T0.3": ArmRole.DEMONSTRATION,
    "fail-T1.3": ArmRole.DEMONSTRATION,
}


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One trajectory with the per-step series the analysis needs."""

    key: str
    arm: str
    role: str
    problem: str
    seed: int
    temperature: float
    top_p: float
    label: TrajectoryClass
    termination: str
    git_commit: str | None
    model_revision: str | None
    dataset_revision: str | None
    entropy: np.ndarray
    tokens: np.ndarray

    @property
    def n(self) -> int:
        return self.entropy.size

    @property
    def testable(self) -> bool:
        """Whether a hypothesis may be tested on the arm this came from."""
        return self.role in TESTABLE_ROLES


def arm_from_traj_id(traj_id: str) -> str:
    """Arm name parsed out of the trajectory id.

    Fallback for the runs written before `arm_id` existed. New runs carry the
    column and must not reach this.
    """
    return traj_id.split("-", 3)[-1].rsplit("-r", 1)[0]


#: Quantisation for the float32 sampler settings, in units of the stored value.
#: 1e-3 is three orders of magnitude coarser than any temperature this study
#: separates and five coarser than the float32 round-trip error.
_GRID: Final[int] = 1000

#: Least permissive first. On a role conflict the survivor keeps the smallest,
#: so a collision can never promote a trajectory into `TESTABLE_ROLES`.
_ROLE_ORDER: Final[dict[str, int]] = {
    str(ArmRole.DEMONSTRATION): 0,
    str(ArmRole.REFERENCE): 1,
    str(ArmRole.HYPOTHESIS): 2,
}


def _grid(value: float) -> int:
    """A float32 sampler setting as an exact, reader-independent key."""
    return round(float(value) * _GRID)


def _identity(record: Trajectory) -> tuple:
    return (record.problem, _grid(record.temperature),
            _grid(record.top_p), record.seed)


def _full_identity(record: Trajectory) -> tuple:
    return _identity(record) + (
        record.git_commit, record.model_revision, record.dataset_revision)


def load_runs(
    runs_root: Path, only: str | None = None
) -> tuple[list[Trajectory], list[str]]:
    """Every untagged run under `runs_root`, with identity enforced.

    `only` restricts to run directories whose name contains it. Labels depend on
    the ceiling the run used, so pooling runs that used different ceilings
    compares labels produced under different rules -- the pilot ran at 8192 and
    the diagnostics at 2048/4096, and they must not be mixed for anything that
    touches a class.

    Returns the surviving trajectories and a human-readable log of what was
    superseded. Raises on an exact repeat of the same experiment, because that is
    an assembly error and silently keeping one copy is what this module exists to
    prevent.
    """
    seen_full: dict[tuple, Trajectory] = {}
    by_experiment: dict[tuple, Trajectory] = {}
    superseded: list[str] = []

    for run in sorted(runs_root.iterdir()):          # names are timestamps
        if not run.is_dir() or run.name.startswith("_"):
            continue
        if only is not None and only not in run.name:
            continue
        if not any(run.glob("trajectories*.parquet")):
            continue
        # A run in flight has no readable shard until its first checkpoint. That
        # is not an error for the analysis; it is a run that has nothing to
        # contribute yet.
        try:
            traj = read_trajectories(run).to_pydict()
            step = read_steps(
                run, columns=["traj_id", "entropy", "token_id"]).to_pydict()
        except FileNotFoundError as exc:
            print(f"  [skip] {run.name}: {exc}")
            continue
        if not traj["traj_id"]:
            continue

        entropy = np.asarray(step["entropy"], dtype=np.float64)
        token_id = np.asarray(step["token_id"])
        has_arm = "arm_id" in traj

        # Index the steps once. Masking with `step_traj == traj_id` per
        # trajectory is O(trajectories x steps) string comparisons -- 400 x 1M on
        # the stage-1 run, which took minutes. Steps are written contiguously per
        # trajectory, so one pass recording each run of equal ids is enough.
        step_traj = np.asarray(step["traj_id"])
        bounds: dict[str, tuple[int, int]] = {}
        if step_traj.size:
            edges = np.flatnonzero(step_traj[1:] != step_traj[:-1]) + 1
            starts = np.concatenate([[0], edges])
            stops = np.concatenate([edges, [step_traj.size]])
            for a, b in zip(starts, stops):
                key = step_traj[a]
                # A trajectory split across checkpoint parts appears as several
                # runs; widen rather than overwrite.
                lo, hi = bounds.get(key, (a, b))
                bounds[key] = (min(lo, a), max(hi, b))

        # The `git_commit` column was declared from the start and populated from
        # none: every row written before this module existed is null. The
        # run-level file has always been correct, so recover from it rather than
        # weakening the identity key to whatever happens to be in the column.
        commit_file = run / "git_commit"
        run_commit = (commit_file.read_text().strip()
                      if commit_file.exists() else None)
        if run_commit in ("", "unknown"):
            run_commit = None

        for i, traj_id in enumerate(traj["traj_id"]):
            arm = traj["arm_id"][i] if has_arm else arm_from_traj_id(traj_id)
            if has_arm:
                role = traj["arm_role"][i]
            elif arm in LEGACY_ARM_ROLES:
                role = str(LEGACY_ARM_ROLES[arm])
            else:
                raise ValueError(
                    f"{run.name}/{traj_id}: arm {arm!r} has no role. The run "
                    "predates the arm_role column and is not in "
                    "LEGACY_ARM_ROLES; add it there rather than guessing."
                )

            lo, hi = bounds[traj_id]
            record = Trajectory(
                key=f"{run.name}/{traj_id}",
                arm=arm,
                role=role,
                problem=traj["problem_id"][i],
                seed=traj["seed"][i],
                temperature=traj["temperature"][i],
                top_p=traj["top_p"][i],
                label=classify(traj["termination"][i], traj["correct"][i],
                               traj["extraction_outcome"][i]),
                termination=traj["termination"][i],
                git_commit=traj["git_commit"][i] or run_commit,
                model_revision=traj["model_revision"][i],
                dataset_revision=traj["dataset_revision"][i],
                entropy=entropy[lo:hi],
                tokens=token_id[lo:hi],
            )

            full = _full_identity(record)
            if full in seen_full:
                raise ValueError(
                    f"the same experiment appears twice:\n"
                    f"  {seen_full[full].key}\n  {record.key}\n"
                    f"identity = {full}\n"
                    "Same arm, problem, seed, code and revisions, so these are "
                    "not two measurements -- they are one, counted twice. Move "
                    "one run directory aside (prefix it) rather than letting "
                    "the loader pick."
                )
            seen_full[full] = record

            experiment = _identity(record)
            if experiment in by_experiment:
                previous = by_experiment[experiment]
                note = ""
                if previous.role != record.role:
                    # Two declarations of intent for one configuration. Neither
                    # is more recent in any meaningful sense -- the later run
                    # simply repeated a cell the earlier one had already used --
                    # so take the reading that permits less.
                    kept = min(previous.role, record.role,
                               key=lambda r: _ROLE_ORDER[str(r)])
                    note = (f"  [role conflict: {previous.arm} is {previous.role},"
                            f" {record.arm} is {record.role}; keeping {kept}]")
                    record = replace(record, role=kept)
                superseded.append(
                    f"{previous.key} ({previous.n} steps, {previous.termination})"
                    f" -> {record.key} ({record.n} steps, {record.termination})"
                    f"{note}")
            by_experiment[experiment] = record

    return list(by_experiment.values()), superseded


def testable(trajectories: list[Trajectory]) -> list[Trajectory]:
    """Trajectories from arms on which a hypothesis may be tested.

    Use this in front of every AUC. An arm that cannot supply both classes
    cannot test H1a, and pooling a demonstration arm into the comparison
    measures the temperature rather than the hypothesis: across all arms the
    H1a AUC reads 0.854, stratified within arm 0.600.
    """
    return [t for t in trajectories if t.testable]


def require_testable(trajectories: list[Trajectory], what: str) -> None:
    """Raise if `trajectories` contains an arm no hypothesis may be tested on."""
    offenders = sorted({f"{t.arm} ({t.role})"
                        for t in trajectories if not t.testable})
    if offenders:
        raise ValueError(
            f"{what} was asked to include non-hypothesis arms: "
            f"{', '.join(offenders)}. Filter with runs.testable() first; "
            "these arms are descriptive and their class counts are "
            "manufactured."
        )
