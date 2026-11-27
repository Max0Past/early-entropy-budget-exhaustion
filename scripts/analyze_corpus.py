#!/usr/bin/env python
"""One reproducible source for every number the project currently quotes.

This script exists because the numbers did not have one. `confound.json`,
`regimes.json` and `verify9e47ab.json` sat in the repository root, were the only
two-run-aware artefacts in the project, and **no file in the repository
produced them**. Meanwhile the committed analysis had drifted: seven scripts
hard-code the stage-1 run directory, three filter on an arm name (`healthy`)
that exists in neither run, and two pool every run and silently double-count 52
duplicate trajectories.

Two design points follow from that history.

* **Runs are arguments, never defaults.** A script that picks a run for you
  picks the wrong one the moment a second run exists, and says nothing.
* **De-duplication matches on `(problem_id, temperature, top_p, seed)`, not on
  `arm_id`.** `runs.py` defines identity as `(arm, problem, seed)`, so renaming
  an arm walks straight past the guard: `demo-T0.3` and the first 52 rows of
  `cold-T0.3` are bit-identical and were counted twice by everything that
  pooled.

Usage:
    python scripts/analyze_corpus.py RUN_DIR [RUN_DIR ...] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cotctl.labeling import classify, tail_period, periodicity_onset  # noqa: E402

# The prefix grid for the early-warning question. K is the cutoff; the feature is
# the mean signal over [K-128, K), so the risk set is trajectories still running
# at K.
PREFIX_GRID: Final[tuple[int, ...]] = (256, 512, 1024)
FEATURE_WINDOW: Final[int] = 128
SIGNALS: Final[tuple[str, ...]] = ("entropy", "margin", "surprisal", "kl_prev", "logit_var")
# An arm contributes to the arm-stratified estimate only with both classes
# present; three of each is the minimum that gives the within-arm rank sum any
# resolution at all.
MIN_PER_CLASS: Final[int] = 3
BOOTSTRAP: Final[int] = 2000
SEED: Final[int] = 20260906
# Detector operating point carried over from the stage-1 grid search.
DETECTOR_LEVEL: Final[float] = 0.05
DETECTOR_PERSISTENCE: Final[int] = 400
SMOOTHING_WINDOW: Final[int] = 24


def read_parquet_glob(run: Path, kind: str, columns: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(run / f"{kind}-*.parquet"))):
        try:
            d = pd.read_parquet(f, columns=columns) if columns else pd.read_parquet(f)
        except Exception:
            continue
        if len(d):
            frames.append(d)
    if not frames:
        raise SystemExit(f"no readable {kind}-*.parquet under {run}")
    return pd.concat(frames, ignore_index=True)


def load_corpus(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load every run, drop duplicates, return (trajectories, steps, report)."""
    tframes, sframes = [], []
    for run in runs:
        t = read_parquet_glob(run, "trajectories")
        t["run"] = run.name
        s = read_parquet_glob(run, "steps",
                              ["traj_id", "step", "token_id", *SIGNALS])
        s["run"] = run.name
        tframes.append(t)
        sframes.append(s)

    tr = pd.concat(tframes, ignore_index=True)
    before = len(tr)

    key = ["problem_id", "temperature", "top_p", "seed"]
    dup_mask = tr.duplicated(subset=key, keep="first")
    dropped = tr.loc[dup_mask, ["run", "traj_id", "arm_id"]].to_dict("records")
    tr = tr[~dup_mask].copy()

    st = pd.concat(sframes, ignore_index=True)
    st = st.merge(tr[["run", "traj_id"]], on=["run", "traj_id"])
    st = st.sort_values(["run", "traj_id", "step"], kind="mergesort")

    tr["_k"] = list(zip(tr.run, tr.traj_id))
    tr["ceiling"] = tr.termination == "max_tokens"
    # classify returns a StrEnum, which pandas stores as its string value.
    tr["cls"] = [str(classify(a, b, c)) for a, b, c in
                 zip(tr.termination, tr.correct, tr.extraction_outcome)]

    report = {
        "runs": [r.name for r in runs],
        "trajectories_before_dedup": before,
        "duplicates_dropped": int(dup_mask.sum()),
        "independent_trajectories": len(tr),
        "steps": int(len(st)),
        "problems": int(tr.problem_id.nunique()),
        "dropped_examples": dropped[:5],
    }
    return tr, st, report


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def auc(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney AUC of x against binary y, with the comparison weight."""
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), 0.0
    r = pd.Series(x).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), float(n1 * n0)


def stratified_auc(values: np.ndarray, labels: np.ndarray, strata: np.ndarray) -> float:
    """Pooled AUC over strata, weighted by each stratum's comparison count."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a, w = auc(values[m], labels[m])
        if not np.isnan(a):
            num += a * w
            den += w
    return num / den if den else float("nan")


def resample_clusters(clusters: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """One bootstrap draw of whole clusters: row indices and, per row, which copy it is.

    A cluster drawn k times contributes k copies; `copy` numbers them so that callers
    can keep the copies apart.
    """
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in np.unique(clusters)}
    keys = list(idx_by_cluster)
    draw = rng.integers(0, len(keys), len(keys))
    pick = np.concatenate([idx_by_cluster[keys[i]] for i in draw])
    copy = np.concatenate([np.full(len(idx_by_cluster[keys[i]]), j) for j, i in enumerate(draw)])
    return pick, copy


def resampled_strata(strata: np.ndarray, clusters: np.ndarray, pick: np.ndarray,
                     copy: np.ndarray) -> np.ndarray:
    """Strata labels for a bootstrap draw.

    When every stratum lies inside one cluster (stratifying by problem), each copy of a
    cluster must be its own stratum. Merging k copies into one stratum would multiply
    its pair count by k**2 instead of k, over-weighting problems that happen to be
    drawn more than once. Strata that span clusters (decoding conditions) keep their
    labels.
    """
    codes = np.unique(strata, return_inverse=True)[1]
    nested = pd.Series(clusters).groupby(codes).nunique().max() == 1
    if not nested:
        return codes[pick]
    return codes[pick] * (copy.max() + 1) + copy


def cluster_ci(values, labels, strata, clusters, rng, n=BOOTSTRAP):
    """Percentile CI, resampling whole clusters. Invariant 9: the cluster is the
    problem, because trajectories attempting the same problem are correlated."""
    draws = []
    for _ in range(n):
        pick, copy = resample_clusters(clusters, rng)
        v = stratified_auc(values[pick], labels[pick],
                           resampled_strata(strata, clusters, pick, copy))
        if not np.isnan(v):
            draws.append(v)
    if not draws:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))


def moving_average(a: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(a, dtype=float)])
    return (c[w:] - c[:-w]) / w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out: dict = {}
    tr, st, report = load_corpus(args.runs)
    out["manifest"] = report

    print("=== 0. corpus ===")
    print(f"  runs                    {', '.join(report['runs'])}")
    print(f"  trajectories loaded     {report['trajectories_before_dedup']}")
    print(f"  duplicates dropped      {report['duplicates_dropped']}"
          "   (matched on problem, temperature, top_p, seed)")
    print(f"  independent             {report['independent_trajectories']}")
    print(f"  steps                   {report['steps']:,}")
    print(f"  problems                {report['problems']}")

    # --- 1. classes -----------------------------------------------------------
    print("\n=== 1. class composition ===")
    counts = tr.cls.value_counts()
    out["classes"] = {k: int(v) for k, v in counts.items()}
    for k, v in counts.items():
        lo, hi = wilson(int(v), len(tr))
        print(f"  {k:20s} {int(v):4d}  {100*v/len(tr):5.1f}%"
              f"  [{100*lo:4.1f}, {100*hi:4.1f}]")

    # --- 2. ceiling by temperature -------------------------------------------
    print("\n=== 2. failure to terminate, by arm ===")
    rows = []
    for (arm, T, tp), g in tr.groupby(["arm_id", "temperature", "top_p"]):
        k, n = int(g.ceiling.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(arm=arm, T=float(T), top_p=float(tp), n=n, ceiling=k,
                         pct=100 * k / n, lo=100 * lo, hi=100 * hi))
    rows.sort(key=lambda r: r["T"])
    out["ceiling_by_arm"] = rows
    print(f"  {'arm':12s} {'T':>5} {'top_p':>6} {'n':>5} {'ceiling':>8}  95% CI")
    for r in rows:
        print(f"  {r['arm']:12s} {r['T']:5.1f} {r['top_p']:6.2f} {r['n']:5d}"
              f" {r['ceiling']:4d} ({r['pct']:4.1f}%)  [{r['lo']:4.1f}, {r['hi']:4.1f}]")

    # --- 3. exact periodicity -------------------------------------------------
    print("\n=== 3. exactly periodic tails ===")
    tok = {k: g.token_id.tolist() for k, g in st.groupby(["run", "traj_id"], sort=False)}
    ceil_of = dict(zip(tr._k, tr.ceiling))
    arm_of = dict(zip(tr._k, tr.arm_id))
    per_rows = []
    for k, ids in tok.items():
        p = tail_period(ids)
        if p is None:
            continue
        per_rows.append(dict(traj=k[1], run=k[0], arm=arm_of[k], ceiling=bool(ceil_of[k]),
                             period=int(p), onset=int(periodicity_onset(ids, p)),
                             length=len(ids)))
    per = pd.DataFrame(per_rows)
    out["periodic"] = {"n": len(per), "n_ceiling": int(per.ceiling.sum()) if len(per) else 0,
                       "rows": per_rows}
    if len(per):
        wasted = int((per.length - per.onset).sum())
        out["periodic"]["wasted_tokens"] = wasted
        print(f"  trajectories with an exactly periodic tail   {len(per)}")
        print(f"    of which reached the ceiling               {int(per.ceiling.sum())}")
        print(f"    period   median {int(per.period.median()):3d}"
              f"   range {int(per.period.min())}-{int(per.period.max())}")
        print(f"    onset    median {int(per.onset.median()):4d}"
              f"  range {int(per.onset.min())}-{int(per.onset.max())}")
        print(f"    tokens emitted after onset                 {wasted:,}"
              f"  ({100*wasted/report['steps']:.1f}% of corpus)")
        print("    by arm: " + ", ".join(
            f"{a} {n}" for a, n in per.groupby('arm').size().items()))

    # --- 4. early-warning AUC -------------------------------------------------
    print("\n=== 4. does an early signal predict failure to terminate ===")
    print("  stratified by ARM holds the sampling condition fixed and varies the problem;")
    print("  stratified by PROBLEM holds the problem fixed and varies the condition.")
    rng = np.random.default_rng(SEED)
    auc_rows = []
    for K in PREFIX_GRID:
        at = tr[tr.n_steps > K].copy()
        win = st[(st.step >= K - FEATURE_WINDOW) & (st.step < K)]
        feats = win.groupby(["run", "traj_id"])[list(SIGNALS)].mean()
        at = at.join(feats, on=["run", "traj_id"])
        at = at.dropna(subset=["entropy"])
        keep = [a for a, g in at.groupby("arm_id")
                if g.ceiling.sum() >= MIN_PER_CLASS and (~g.ceiling).sum() >= MIN_PER_CLASS]
        sub = at[at.arm_id.isin(keep)]
        y = sub.ceiling.to_numpy().astype(int)
        print(f"\n  K = {K}:  risk set {len(at)}, after arm filter {len(sub)}"
              f" ({len(at) - len(sub)} dropped from {len(set(at.arm_id) - set(keep))} arm(s)),"
              f" events {int(y.sum())}")
        for sig in SIGNALS:
            v = sub[sig].to_numpy(dtype=float)
            if np.isnan(v).all():
                continue
            a_arm = stratified_auc(v, y, sub.arm_id.to_numpy())
            a_prb = stratified_auc(v, y, sub.problem_id.to_numpy())
            lo_a, hi_a = cluster_ci(v, y, sub.arm_id.to_numpy(),
                                    sub.problem_id.to_numpy(), rng)
            lo_p, hi_p = cluster_ci(v, y, sub.problem_id.to_numpy(),
                                    sub.problem_id.to_numpy(), rng)
            auc_rows.append(dict(K=K, signal=sig, by_arm=a_arm, by_arm_lo=lo_a,
                                 by_arm_hi=hi_a, by_problem=a_prb,
                                 by_problem_lo=lo_p, by_problem_hi=hi_p,
                                 n=len(sub), events=int(y.sum())))
            print(f"    {sig:11s} by arm {a_arm:.3f} [{lo_a:.3f}, {hi_a:.3f}]"
                  f"   by problem {a_prb:.3f} [{lo_p:.3f}, {hi_p:.3f}]")
    out["auc"] = auc_rows

    # --- 5. tail entropy at the ceiling ---------------------------------------
    print("\n=== 5. entropy change at the ceiling, by arm ===")
    ent = {k: g.entropy.to_numpy() for k, g in st.groupby(["run", "traj_id"], sort=False)}
    d_rows = []
    for _, r in tr[tr.ceiling].iterrows():
        h = ent[r._k]
        if len(h) < 1400:
            continue
        d_rows.append(dict(arm=r.arm_id, T=float(r.temperature),
                           dH=float(np.median(h[-1000:]) - np.median(h[:256]))))
    dd = pd.DataFrame(d_rows)
    tail_rows = []
    for (arm, T), g in dd.groupby(["arm", "T"]):
        tail_rows.append(dict(arm=arm, T=float(T), n=len(g),
                              dH_median=float(g.dH.median()),
                              n_positive=int((g.dH > 0).sum())))
    tail_rows.sort(key=lambda r: r["T"])
    out["tail_entropy"] = tail_rows
    print(f"  {'arm':12s} {'T':>5} {'n':>4} {'median dH':>10} {'rising':>8}")
    for r in tail_rows:
        print(f"  {r['arm']:12s} {r['T']:5.1f} {r['n']:4d} {r['dH_median']:10.3f}"
              f" {r['n_positive']:4d}/{r['n']}")

    # --- 6. the stopping detector ---------------------------------------------
    print(f"\n=== 6. detector: ma{SMOOTHING_WINDOW}(entropy) < {DETECTOR_LEVEL}"
          f" for {DETECTOR_PERSISTENCE} consecutive steps ===")
    fired_ceiling, fired_other, saved = [], [], 0
    for k, h in ent.items():
        if len(h) < SMOOTHING_WINDOW + DETECTOR_PERSISTENCE:
            continue
        below = moving_average(h, SMOOTHING_WINDOW) < DETECTOR_LEVEL
        c = np.concatenate([[0], np.cumsum(below)])
        runs_ok = np.flatnonzero(c[DETECTOR_PERSISTENCE:] - c[:-DETECTOR_PERSISTENCE]
                                 == DETECTOR_PERSISTENCE)
        if not runs_ok.size:
            continue
        t_fire = int(runs_ok[0]) + DETECTOR_PERSISTENCE + SMOOTHING_WINDOW - 1
        if ceil_of[k]:
            fired_ceiling.append(t_fire)
            saved += len(h) - t_fire
        else:
            fired_other.append(t_fire)
    n_ceiling = int(tr.ceiling.sum())
    out["detector"] = {
        "level": DETECTOR_LEVEL, "persistence": DETECTOR_PERSISTENCE,
        "fired_on_ceiling": len(fired_ceiling), "ceiling_total": n_ceiling,
        "false_alarms": len(fired_other), "non_ceiling_total": int((~tr.ceiling).sum()),
        "tokens_saved": int(saved), "corpus_steps": report["steps"],
        "median_fire_step": int(np.median(fired_ceiling)) if fired_ceiling else None,
    }
    print(f"  fires on ceiling trajectories   {len(fired_ceiling)} / {n_ceiling}")
    print(f"  false alarms                    {len(fired_other)} /"
          f" {int((~tr.ceiling).sum())}")
    if fired_ceiling:
        print(f"  median firing step              {int(np.median(fired_ceiling))}")
        print(f"  tokens saved by stopping there  {saved:,}"
              f"  ({100*saved/report['steps']:.2f}% of corpus)")

    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
