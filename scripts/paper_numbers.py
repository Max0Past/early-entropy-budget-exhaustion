#!/usr/bin/env python
"""Numbers quoted in the paper that `analyze_corpus.py` does not print.

`analyze_corpus.py` stays the single entry point for the corpus, the early-warning
AUROCs, the tail-entropy change and the stopping detector. This script adds the rest,
on the same de-duplicated corpus and with the same helpers:

  7. instrument check at the first decoding step (invariant vs tempered reading);
  8. how well the stored top-k logits reproduce the full-vocabulary entropy;
  9. problem difficulty and early entropy (problem-level coupling);
 10. early-warning AUROC after removing each problem's mean (residual ranking);
 11. how many events each target definition yields;
 12. an exploratory sweep of a persistence detector against located onsets;
 13. the same detector against a period-agnostic repetition marker, which dates the
     onset of the 47 exactly periodic failures without the exact-periodicity lag;
 14. details the text quotes: strata behind the within-problem estimate, and how the
     tails of exhausted and terminated trajectories differ;
 15. exploratory baselines that see no trajectory (entropy at step 0, temperature) and
     paired bootstrap differences between the two stratifications.

Usage:
    python scripts/paper_numbers.py RUN_DIR [RUN_DIR ...] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_corpus import (  # noqa: E402
    BOOTSTRAP, FEATURE_WINDOW, MIN_PER_CLASS, PREFIX_GRID, SEED, SMOOTHING_WINDOW,
    cluster_ci, load_corpus, moving_average, resample_clusters, resampled_strata,
    stratified_auc, wilson,
)
from cotctl.labeling import (  # noqa: E402
    PERIODIC_TAIL, find_onset, periodicity_onset, tail_period,
)

PRIMARY_K: Final[int] = 256
NEAR_ZERO_TAIL: Final[float] = 0.05   # nats; same level as the stopping detector
TAIL_STEPS: Final[int] = 1000
DETECTOR_LEVELS: Final[tuple[float, ...]] = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
DETECTOR_PERSISTENCE: Final[tuple[int, ...]] = (16, 32, 64, 128, 256, 400, 800)
# Section 13. The marker reads token ids only, never the entropy it is compared with.
MARKER_NGRAM: Final[int] = 8          # same n as the repetition marker in labeling.py
MARKER_WINDOWS: Final[tuple[int, ...]] = (128, 256)
MARKER_LEVELS: Final[tuple[float, ...]] = (0.05, 0.10, 0.25)
MARKER_PRIMARY: Final[tuple[int, float]] = (256, 0.05)   # the latest-firing variant
FALSE_ALARM_CAPS: Final[tuple[float, ...]] = (0.01, 0.05, 0.10)


def entropy_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Entropy (nats) of softmax(logits / T) along the last axis."""
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=-1)


def folded_auc(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney AUROC folded to [0.5, 1]: 0.5 = indistinguishable."""
    ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    u = ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2
    auc = u / (len(a) * len(b))
    return float(max(auc, 1 - auc))


def read_steps(runs: list[Path], columns: list[str], expr=None) -> pd.DataFrame:
    """Read step rows from every run, tagging each with its run name."""
    frames = []
    for run in runs:
        files = sorted(run.glob("steps-*.parquet"))
        t = ds.dataset([str(f) for f in files], format="parquet").to_table(
            columns=columns, filter=expr)
        d = t.to_pandas()
        d["run"] = run.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def first_step_check(runs: list[Path], tr: pd.DataFrame) -> dict:
    """Section 7: at step 0 every arm on a problem sees the same logits."""
    s0 = read_steps(runs, ["traj_id", "step", "entropy", "topk_logits"],
                    ds.field("step") == 0)
    s0 = s0.merge(tr[["run", "traj_id", "problem_id", "arm_id", "temperature"]],
                  on=["run", "traj_id"])
    logits = np.stack(s0.topk_logits.to_numpy())
    s0["tempered"] = [float(entropy_from_logits(z, t))
                      for z, t in zip(logits, s0.temperature)]

    spread = s0.groupby("problem_id").entropy.agg(lambda v: v.max() - v.min())
    multi = s0.groupby("problem_id").arm_id.nunique() >= 2
    by_arm = (s0.groupby(["arm_id", "temperature"])
                .agg(n=("entropy", "size"), invariant=("entropy", "mean"),
                     tempered=("tempered", "mean"))
                .reset_index().sort_values("temperature"))

    # Pairs of sampling temperatures (the two T = 0.3 arms form one condition).
    pairs = []
    temps = sorted(s0.temperature.round(3).unique())
    for ta, tb in itertools.combinations(temps, 2):
        x = s0.loc[s0.temperature.round(3) == ta, "tempered"].to_numpy()
        y = s0.loc[s0.temperature.round(3) == tb, "tempered"].to_numpy()
        pairs.append(dict(a=float(ta), b=float(tb), auc=folded_auc(x, y)))
    pair_auc = np.array([p["auc"] for p in pairs])

    print("\n=== 7. first decoding step: invariant vs tempered reading ===")
    print(f"  problems with >= 2 arms          {int(multi.sum())}")
    print(f"  invariant spread across arms     max {spread[multi].max():.2e}"
          f"  median {spread[multi].median():.2e} nats")
    print(f"  {'arm':12s} {'T':>4} {'n':>4} {'invariant':>10} {'tempered':>9}")
    for r in by_arm.itertuples(index=False):
        print(f"  {r.arm_id:12s} {r.temperature:4.1f} {r.n:4d}"
              f" {r.invariant:10.4f} {r.tempered:9.3f}")
    print(f"  arm means, invariant: range {by_arm.invariant.max() - by_arm.invariant.min():.4f}"
          " nats (arms attempt different problem sets)")
    print(f"  arm means, tempered:  range {by_arm.tempered.max() - by_arm.tempered.min():.3f} nats")
    print(f"  tempered reading separates two arms: AUROC {pair_auc.min():.3f}-{pair_auc.max():.3f},"
          f" mean {pair_auc.mean():.3f} over {len(pairs)} pairs of temperatures")
    return {
        "n_problems_multi_arm": int(multi.sum()),
        "invariant_spread_max": float(spread[multi].max()),
        "invariant_spread_median": float(spread[multi].median()),
        "by_arm": by_arm.to_dict("records"),
        "pairwise_auc": pairs,
        "pairwise_auc_min": float(pair_auc.min()),
        "pairwise_auc_max": float(pair_auc.max()),
        "pairwise_auc_mean": float(pair_auc.mean()),
    }


def topk_recompute_check(runs: list[Path], grid: list[float], rng) -> dict:
    """Section 8: entropy from the 32 stored logits vs the stored full-vocab value."""
    s0 = read_steps(runs, ["topk_logits", "entropy_grid"], ds.field("step") == 0)
    l0, f0 = np.stack(s0.topk_logits.to_numpy()), np.stack(s0.entropy_grid.to_numpy())
    first = {temp: float(np.abs(entropy_from_logits(l0, temp) - f0[:, j]).max())
             for j, temp in enumerate(grid)}
    print("\n=== 8. entropy recomputed from the 32 stored logits ===")
    print("  first step, max |top-32 minus full vocabulary| over all trajectories:")
    print("   " + "  ".join(f"T={t}: {e:.1e}" for t, e in first.items()))
    st = read_steps(runs, ["topk_logits", "entropy_grid"])
    pick = rng.choice(len(st), size=min(200_000, len(st)), replace=False)
    logits = np.stack(st.topk_logits.to_numpy()[pick])
    full = np.stack(st.entropy_grid.to_numpy()[pick])
    rows = []
    print(f"  {len(pick):,} random steps; |top-32 minus full vocabulary|, nats")
    for j, temp in enumerate(grid):
        err = np.abs(entropy_from_logits(logits, temp) - full[:, j])
        rows.append(dict(T=temp, median=float(np.median(err)),
                         p99=float(np.percentile(err, 99)), max=float(err.max())))
        print(f"  T={temp:3.1f}  median {np.median(err):.1e}  p99 {np.percentile(err, 99):.1e}"
              f"  max {err.max():.2e}")
    return {"first_step_max": first, "n_steps": int(len(pick)), "rows": rows}


def early_features(tr: pd.DataFrame, st: pd.DataFrame, k: int) -> pd.DataFrame:
    """Risk set at cutoff k with the mean entropy over [k-128, k), as in analyze_corpus."""
    at = tr[tr.n_steps > k].copy()
    win = st[(st.step >= k - FEATURE_WINDOW) & (st.step < k)]
    feats = win.groupby(["run", "traj_id"])["entropy"].mean()
    return at.join(feats, on=["run", "traj_id"]).dropna(subset=["entropy"])


def difficulty_coupling(tr: pd.DataFrame, st: pd.DataFrame) -> dict:
    """Section 9: problems that fail more often also start with higher entropy."""
    at = early_features(tr, st, PRIMARY_K)
    per = at.groupby("problem_id").agg(h=("entropy", "mean"))
    per["fails"] = tr.groupby("problem_id").ceiling.sum()
    rho, p = spearmanr(per.fails, per.h)
    print("\n=== 9. problem difficulty and early entropy ===")
    print(f"  {len(per)} problems; failing trajectories per problem vs mean entropy"
          f" over steps {PRIMARY_K - FEATURE_WINDOW}-{PRIMARY_K - 1}")
    print(f"  Spearman rho {rho:.3f}, p = {p:.1e}")
    print(f"  problems with >= 1 failure: {int((per.fails > 0).sum())}")
    return {"n_problems": int(len(per)), "rho": float(rho), "p": float(p),
            "problems_with_failure": int((per.fails > 0).sum())}


def residual_ranking(tr: pd.DataFrame, st: pd.DataFrame, rng) -> dict:
    """Section 10: arm-stratified AUROC after subtracting each problem's mean."""
    at = early_features(tr, st, PRIMARY_K)
    keep = [a for a, g in at.groupby("arm_id")
            if g.ceiling.sum() >= MIN_PER_CLASS and (~g.ceiling).sum() >= MIN_PER_CLASS]
    sub = at[at.arm_id.isin(keep)].copy()
    sub["resid"] = sub.entropy - sub.groupby("problem_id").entropy.transform("mean")
    y = sub.ceiling.to_numpy().astype(int)
    v = sub.resid.to_numpy()
    a = stratified_auc(v, y, sub.arm_id.to_numpy())
    lo, hi = cluster_ci(v, y, sub.arm_id.to_numpy(), sub.problem_id.to_numpy(), rng, BOOTSTRAP)
    print("\n=== 10. residual ranking (entropy minus its problem mean), by arm ===")
    print(f"  K={PRIMARY_K}: {len(sub)} trajectories, {int(y.sum())} events:"
          f" AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]")
    return {"n": int(len(sub)), "events": int(y.sum()), "auc": a, "lo": lo, "hi": hi}


def onsets(tr: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Per trajectory: n-gram repetition onset, exact-periodicity onset, tail entropy."""
    rows = []
    for (run, traj), g in st.groupby(["run", "traj_id"], sort=False):
        ids = g.token_id.tolist()
        period = tail_period(ids)
        rows.append(dict(run=run, traj_id=traj,
                         rep_onset=find_onset(ids),
                         per_onset=periodicity_onset(ids, period) if period else None,
                         tail_h=float(np.median(g.entropy.to_numpy()[-TAIL_STEPS:]))))
    return tr.merge(pd.DataFrame(rows), on=["run", "traj_id"])


def event_definitions(ev: pd.DataFrame) -> dict:
    """Section 11: the same corpus under four target definitions."""
    n = len(ev)
    defs = {
        "repetition_marker": int(ev.rep_onset.notna().sum()),
        "exact_periodic_tail": int(ev.per_onset.notna().sum()),
        "ceiling": int(ev.ceiling.sum()),
        "ceiling_and_near_zero_tail": int((ev.ceiling & (ev.tail_h < NEAR_ZERO_TAIL)).sum()),
    }
    masks = {
        "repetition_marker": ev.rep_onset.notna(),
        "exact_periodic_tail": ev.per_onset.notna(),
        "ceiling": ev.ceiling,
        "ceiling_and_near_zero_tail": ev.ceiling & (ev.tail_h < NEAR_ZERO_TAIL),
    }
    # Trajectories of one problem are dependent, so the rate interval resamples whole
    # problems. A generator of its own keeps the shared stream of later sections intact.
    rng = np.random.default_rng(SEED)
    problems = ev.problem_id.to_numpy()
    uniq = np.unique(problems)
    rows_of = [np.flatnonzero(problems == q) for q in uniq]
    draws = [np.concatenate([rows_of[i] for i in rng.integers(0, len(uniq), len(uniq))])
             for _ in range(BOOTSTRAP)]
    print("\n=== 11. events under each target definition ===")
    print("  rate intervals: Wilson (independent trajectories) and cluster bootstrap by problem")
    cis = {}
    for k, v in defs.items():
        lo, hi = wilson(v, n)
        m = masks[k].to_numpy()
        clo, chi = np.percentile([m[d].mean() for d in draws], [2.5, 97.5])
        cis[k] = (float(clo), float(chi))
        print(f"  {k:28s} {v:4d} / {n}   Wilson [{100*lo:.2f}, {100*hi:.2f}]%"
              f"   cluster [{100*clo:.2f}, {100*chi:.2f}]%")
    rep = ev[ev.rep_onset.notna()]
    print(f"  repetition-marker events: {int(rep.ceiling.sum())}/{len(rep)} at the ceiling,"
          f" temperatures {sorted(set(rep.temperature.round(2)))},"
          f" {rep.problem_id.nunique()} problems,"
          f" extraction outcomes {rep.extraction_outcome.value_counts().to_dict()}")
    print(f"  ceiling events with an exactly periodic tail: "
          f"{int((ev.ceiling & ev.per_onset.notna()).sum())}")
    return {"counts": defs, "cluster_ci": cis}


def first_fire(h: np.ndarray, level: float, persistence: int) -> int | None:
    """Step at which ma24(h) has stayed below `level` for `persistence` steps."""
    if len(h) < SMOOTHING_WINDOW + persistence:
        return None
    below = moving_average(h, SMOOTHING_WINDOW) < level
    c = np.concatenate([[0], np.cumsum(below)])
    ok = np.flatnonzero(c[persistence:] - c[:-persistence] == persistence)
    return int(ok[0]) + persistence + SMOOTHING_WINDOW - 1 if ok.size else None


def longest_run_below(h: np.ndarray, level: float, start: int = 0) -> int:
    """Longest run of consecutive smoothed readings below `level` from `start` on."""
    if len(h) < SMOOTHING_WINDOW:
        return 0
    below = moving_average(h, SMOOTHING_WINDOW)[start:] < level
    best = cur = 0
    for b in below:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def detector_sweep(ev: pd.DataFrame, st: pd.DataFrame) -> dict:
    """Section 12, exploratory: does a persistence detector fire before onset?

    Targets are trajectories with a located onset (exact periodicity, and separately the
    n-gram repetition marker). The false-alarm set is every trajectory that terminated
    with a scorable answer (healthy or semantic error).
    """
    ent = {k: g.entropy.to_numpy() for k, g in st.groupby(["run", "traj_id"], sort=False)}
    neg = ev[ev.cls.isin(["healthy", "semantic_error"])]
    targets = {"periodic": ev[ev.per_onset.notna()], "repetition": ev[ev.rep_onset.notna()]}
    onset_col = {"periodic": "per_onset", "repetition": "rep_onset"}
    rows = []
    for level, pers in itertools.product(DETECTOR_LEVELS, DETECTOR_PERSISTENCE):
        fa = sum(first_fire(ent[(r.run, r.traj_id)], level, pers) is not None
                 for r in neg.itertuples())
        for name, tg in targets.items():
            leads = []
            for r in tg.itertuples():
                t = first_fire(ent[(r.run, r.traj_id)], level, pers)
                if t is not None:
                    leads.append(t - getattr(r, onset_col[name]))
            leads = np.array(leads)
            rows.append(dict(target=name, level=level, persistence=pers,
                             n_targets=len(tg), fired=len(leads),
                             before_onset=int((leads < 0).sum()) if len(leads) else 0,
                             median_lead=float(np.median(leads)) if len(leads) else None,
                             false_alarms=fa, n_negative=len(neg)))
    sweep = pd.DataFrame(rows)

    print("\n=== 12. persistence detector vs located onsets (exploratory) ===")
    print(f"  false-alarm set: {len(neg)} terminated trajectories with a scorable answer")
    for name, g in sweep.groupby("target"):
        n_t = int(g.n_targets.iloc[0])
        print(f"  target = {name} ({n_t} trajectories)")
        zero = g[g.false_alarms == 0].sort_values(["fired", "median_lead"],
                                                  ascending=[False, True])
        if len(zero):
            z = zero.iloc[0]
            print(f"    best zero-false-alarm setting: level {z.level}, persistence"
                  f" {z.persistence}: fires on {z.fired}/{n_t}, {z.before_onset} before onset,"
                  f" median lead {z.median_lead:+.0f} steps")
        for cap in (0.01, 0.05, 0.10):
            ok = g[g.false_alarms <= cap * len(neg)]
            best = ok.before_onset.max() if len(ok) else 0
            print(f"    false alarms <= {int(cap*100):2d}%: at most {best}/{n_t} fire before onset")
        anyb = g[g.before_onset > 0]
        if len(anyb):
            c = anyb.sort_values("false_alarms").iloc[0]
            print(f"    cheapest anticipatory setting: {c.before_onset}/{n_t} before onset at"
                  f" {100*c.false_alarms/len(neg):.0f}% false alarms")
            m = g[g.before_onset == g.before_onset.max()].sort_values("false_alarms").iloc[0]
            print(f"    most fires before onset: {m.before_onset}/{n_t}, at the cheapest"
                  f" {100*m.false_alarms/len(neg):.0f}% false alarms")

    runs_neg = np.array([longest_run_below(ent[(r.run, r.traj_id)], NEAR_ZERO_TAIL)
                         for r in neg.itertuples()])
    print(f"  longest run of ma24 < {NEAR_ZERO_TAIL} in terminated trajectories:"
          f" max {runs_neg.max()}, 99th pct {np.percentile(runs_neg, 99):.0f}")
    post = {}
    for name, tg in targets.items():
        r_post = np.array([longest_run_below(ent[(r.run, r.traj_id)], NEAR_ZERO_TAIL,
                                             int(getattr(r, onset_col[name])))
                           for r in tg.itertuples()])
        post[name] = {"min": int(r_post.min()), "median": float(np.median(r_post))}
        print(f"  same after a {name} onset: min {r_post.min()}, median {np.median(r_post):.0f}")
    return {"sweep": sweep.to_dict("records"),
            "runs_negative_max": int(runs_neg.max()),
            "runs_negative_p99": float(np.percentile(runs_neg, 99)),
            "runs_post_onset": post}


def new_ngram_flags(token_ids: np.ndarray, n: int = MARKER_NGRAM) -> np.ndarray:
    """1 where the n-gram ending at this token has not occurred earlier in the trajectory.

    Unlike the within-window repetition rate behind `find_onset`, the comparison is
    against the whole history, so a cycle of any period registers once it repeats.
    """
    flags = np.zeros(len(token_ids))
    seen: set[tuple[int, ...]] = set()
    ids = token_ids.tolist()
    for i in range(n - 1, len(ids)):
        gram = tuple(ids[i - n + 1:i + 1])
        flags[i] = gram not in seen
        seen.add(gram)
    return flags


def repetition_marker(flags: np.ndarray, window: int, level: float) -> int | None:
    """First step at which at most `level` of the last `window` n-grams are new.

    Causal: the value at step t uses tokens up to t only, like the detector.
    """
    if len(flags) < window:
        return None
    c = np.concatenate([[0], np.cumsum(flags)])
    share = (c[window:] - c[:-window]) / window
    hit = np.flatnonzero(share <= level)
    return int(hit[0]) + window - 1 if hit.size else None


def marker_vs_detector(ev: pd.DataFrame, st: pd.DataFrame) -> dict:
    """Section 13, exploratory: does the detector fire before the text starts repeating?

    Section 12 dates the onset of the 47 periodic failures by exact periodicity. That
    date is late whenever the model loops approximately, or in a different cycle, before
    settling into the final one. Here the onset is the first step at which a trailing
    window of the text consists almost entirely of 8-grams seen before in the same
    trajectory, at six (window, level) settings.
    """
    ent = {k: g.entropy.to_numpy() for k, g in st.groupby(["run", "traj_id"], sort=False)}
    ids = {k: g.token_id.to_numpy() for k, g in st.groupby(["run", "traj_id"], sort=False)}
    neg = ev[ev.cls.isin(["healthy", "semantic_error"])]
    per = ev[ev.per_onset.notna()]
    other = ev[ev.ceiling & ev.per_onset.isna()]
    rep = ev[ev.rep_onset.notna()]
    key = lambda r: (r.run, r.traj_id)  # noqa: E731
    flags = {key(r): new_ngram_flags(ids[key(r)])
             for r in pd.concat([neg, per, other]).itertuples()}

    fires = {}
    for level, pers in itertools.product(DETECTOR_LEVELS, DETECTOR_PERSISTENCE):
        fa = sum(first_fire(ent[key(r)], level, pers) is not None for r in neg.itertuples())
        fires[(level, pers)] = (fa, {key(r): first_fire(ent[key(r)], level, pers)
                                     for r in per.itertuples()})

    print("\n=== 13. persistence detector vs a period-agnostic repetition marker (exploratory) ===")
    print(f"  marker: at most q of the last W {MARKER_NGRAM}-grams are new to the trajectory")
    print(f"  {'W':>4} {'q':>5}  marker on: periodic  other ceiling  terminated"
          f"   before exact onset   detector before marker")
    rows = []
    for window, lvl in itertools.product(MARKER_WINDOWS, MARKER_LEVELS):
        mark = {k: repetition_marker(f, window, lvl) for k, f in flags.items()}
        on_neg = sum(mark[key(r)] is not None for r in neg.itertuples())
        on_per = sum(mark[key(r)] is not None for r in per.itertuples())
        on_other = sum(mark[key(r)] is not None for r in other.itertuples())
        early = np.array([r.per_onset - mark[key(r)] for r in per.itertuples()
                          if mark[key(r)] is not None])
        vs_rep = [mark.get(key(r)) for r in rep.itertuples()]
        vs_rep = [int(m - r.rep_onset) for m, r in zip(vs_rep, rep.itertuples()) if m is not None]
        settings = []
        for (level, pers), (fa, fire) in fires.items():
            ahead = [(fire[k] - mark[k]) for k in fire if fire[k] is not None and mark[k] is not None]
            settings.append(dict(level=level, persistence=pers, false_alarms=fa,
                                 fired=len(ahead), before_marker=int(sum(a < 0 for a in ahead)),
                                 median_lead=float(np.median(ahead)) if ahead else None))
        sdf = pd.DataFrame(settings)
        best = {cap: int(sdf[sdf.false_alarms <= cap * len(neg)].before_marker.max())
                for cap in FALSE_ALARM_CAPS}
        anyb = sdf[sdf.before_marker > 0].sort_values("false_alarms")
        cheapest = anyb.iloc[0].to_dict() if len(anyb) else None
        print(f"  {window:4d} {lvl:5.2f}  {on_per:9d}/{len(per)} {on_other:9d}/{len(other)}"
              f" {on_neg:6d}/{len(neg)}   median {np.median(early):5.0f} steps"
              f"     " + " ".join(f"{best[c]}/{len(per)}@{int(100*c)}%" for c in FALSE_ALARM_CAPS))
        rows.append(dict(window=window, level=lvl, on_periodic=on_per, on_other_ceiling=on_other,
                         on_terminated=on_neg, ahead_of_exact_median=float(np.median(early)),
                         ahead_of_exact_min=int(early.min()), minus_find_onset=vs_rep,
                         detector_before_marker=best, cheapest_anticipation=cheapest,
                         settings=settings))
    for r in rows:
        c = r["cheapest_anticipation"]
        print(f"  W={r['window']} q={r['level']:.2f}: cheapest setting with any fire before the marker:"
              + (f" {int(c['before_marker'])}/{len(per)} at {100*c['false_alarms']/len(neg):.1f}% false"
                 f" alarms (level {c['level']}, persistence {c['persistence']:.0f})" if c else " none")
              + f"; marker minus find_onset on the {len(rep)} marker events: {r['minus_find_onset']}")

    w, q = MARKER_PRIMARY
    mark = {k: repetition_marker(f, w, q) for k, f in flags.items()}
    fa, fire = fires[(0.5, 400)]
    early = [k for k in fire if fire[k] is not None and fire[k] < ev.set_index(["run", "traj_id"]).per_onset[k]]
    after_marker = sum(fire[k] >= mark[k] for k in early)
    print(f"  level 0.5 / 400 ({100*fa/len(neg):.1f}% false alarms): {len(early)} fires precede the"
          f" exact onset, {after_marker} of them follow the marker (W={w}, q={q})")
    share_at_fa = []
    for r in neg.itertuples():
        t = first_fire(ent[key(r)], 0.5, 400)
        if t is not None and t >= w - 1:
            share_at_fa.append(flags[key(r)][t - w + 1:t + 1].mean())
    print(f"  false alarms at that setting: share of new {MARKER_NGRAM}-grams in the last {w}"
          f" tokens, median {np.median(share_at_fa):.2f} (repeating text would be near 0)")
    return {"markers": rows, "primary": {"window": w, "level": q},
            "level_0.5_pers_400": {"false_alarms": fa, "before_exact": len(early),
                                   "after_marker": int(after_marker),
                                   "false_alarm_new_share_median": float(np.median(share_at_fa))}}


def text_details(tr: pd.DataFrame, st: pd.DataFrame) -> dict:
    """Section 14: details the text quotes that the sections above do not print."""
    print("\n=== 14. details quoted in the text ===")
    strata = []
    for k in PREFIX_GRID:
        at = early_features(tr, st, k)
        keep = [a for a, g in at.groupby("arm_id")
                if g.ceiling.sum() >= MIN_PER_CLASS and (~g.ceiling).sum() >= MIN_PER_CLASS]
        sub = at[at.arm_id.isin(keep)]
        both = sub.groupby("problem_id").ceiling.agg(lambda c: bool(c.any() and (~c).any()))
        n_traj = int(sub.problem_id.isin(both[both].index).sum())
        strata.append(dict(K=k, n=len(sub), problems=int(both.sum()), trajectories=n_traj))
        print(f"  K={k}: {len(sub)} trajectories; the within-problem estimate uses"
              f" {int(both.sum())} problems with both outcomes ({n_traj} trajectories)")

    ids = {k: g.token_id.to_numpy() for k, g in st.groupby(["run", "traj_id"], sort=False)}
    rows = []
    for r in tr.itertuples():
        a = ids[(r.run, r.traj_id)]
        if len(a) < PERIODIC_TAIL:
            continue
        rows.append(dict(arm=r.arm_id, T=float(r.temperature), ceiling=bool(r.ceiling),
                         periodic=tail_period(a.tolist()) is not None,
                         new=float(new_ngram_flags(a)[-PERIODIC_TAIL:].mean())))
    tails = pd.DataFrame(rows)
    term = tails[~tails.ceiling]
    print(f"  last {PERIODIC_TAIL} tokens, share of {MARKER_NGRAM}-grams new to the trajectory:")
    print(f"    terminated (>= {PERIODIC_TAIL} tokens, n={len(term)}): median {term.new.median():.3f},"
          f" exactly periodic {int(term.periodic.sum())}")
    by_arm = []
    for (arm, t), g in tails[tails.ceiling].groupby(["arm", "T"]):
        by_arm.append(dict(arm=arm, T=t, n=len(g), periodic=int(g.periodic.sum()),
                           new_median=float(g.new.median())))
    for r in sorted(by_arm, key=lambda r: r["T"]):
        print(f"    exhausted, {r['arm']:10s} T={r['T']:.1f}: n={r['n']:2d}, median {r['new_median']:.3f},"
              f" exactly periodic {r['periodic']}")
    answers = tr[tr.ceiling].extraction_outcome.value_counts().to_dict()
    print(f"  exhausted trajectories by extraction outcome: {answers},"
          f" correct {int(tr[tr.ceiling].correct.sum())}")
    return {"strata": strata, "terminated_tail": {"n": len(term), "new_median": float(term.new.median()),
            "periodic": int(term.periodic.sum())}, "exhausted_tail": by_arm,
            "exhausted_answers": {k: int(v) for k, v in answers.items()}}


def paired_ci(stats, labels, clusters, rng, n=BOOTSTRAP):
    """Percentile CI of a difference of stratified AUROCs, on shared bootstrap draws.

    `stats` is a pair of (values, strata); each draw resamples whole problems once and
    evaluates both, so the interval reflects the pairing.
    """
    diffs = []
    for _ in range(n):
        pick, copy = resample_clusters(clusters, rng)
        a = [stratified_auc(v[pick], labels[pick], resampled_strata(s, clusters, pick, copy))
             for v, s in stats]
        if not np.isnan(a[0]) and not np.isnan(a[1]):
            diffs.append(a[0] - a[1])
    return tuple(float(x) for x in np.percentile(diffs, [2.5, 97.5]))


def baselines(tr: pd.DataFrame, st: pd.DataFrame, rng) -> dict:
    """Section 15, exploratory: what early entropy adds over signals that see no trajectory.

    Entropy at step 0 is read before any token is sampled, so it is a property of the
    prompt; its sign (lower initial entropy, higher risk) was chosen after looking at the
    data. Temperature is a property of the decoding condition. Neither sees the trajectory.
    """
    at = early_features(tr, st, PRIMARY_K)
    h0 = st[st.step == 0].set_index(["run", "traj_id"]).entropy
    at["h0"] = h0.reindex(pd.MultiIndex.from_arrays([at.run, at.traj_id])).to_numpy()
    keep = [a for a, g in at.groupby("arm_id")
            if g.ceiling.sum() >= MIN_PER_CLASS and (~g.ceiling).sum() >= MIN_PER_CLASS]
    sub = at[at.arm_id.isin(keep)]
    y = sub.ceiling.to_numpy().astype(int)
    arm, prob = sub.arm_id.to_numpy(), sub.problem_id.to_numpy()
    scores = {"mean entropy, steps 128-255": sub.entropy.to_numpy(),
              "minus entropy at step 0": -sub.h0.to_numpy(),
              "minus temperature": -sub.temperature.to_numpy().astype(float)}
    print("\n=== 15. baselines that see no trajectory, and paired differences (exploratory) ===")
    print(f"  K={PRIMARY_K}: {len(sub)} trajectories, {int(y.sum())} exhausted")
    rows = {}
    for name, v in scores.items():
        a_c, a_p = stratified_auc(v, y, arm), stratified_auc(v, y, prob)
        rows[name] = {"within_condition": a_c, "within_problem": a_p}
        print(f"    {name:28s} within condition {a_c:.3f}   within problem {a_p:.3f}")
    e, h = scores["mean entropy, steps 128-255"], scores["minus entropy at step 0"]
    d1 = rows["mean entropy, steps 128-255"]["within_condition"] - rows["mean entropy, steps 128-255"]["within_problem"]
    lo1, hi1 = paired_ci([(e, arm), (e, prob)], y, prob, rng)
    d2 = rows["mean entropy, steps 128-255"]["within_condition"] - rows["minus entropy at step 0"]["within_condition"]
    lo2, hi2 = paired_ci([(e, arm), (h, arm)], y, prob, rng)
    print(f"  entropy, within condition minus within problem: {d1:+.3f} [{lo1:+.3f}, {hi1:+.3f}]")
    print(f"  within condition, entropy minus step-0 baseline: {d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")
    pairs = opposite = ties = 0
    for a in np.unique(arm):
        m = arm == a
        pos, neg = m & (y == 1), m & (y == 0)
        de = e[pos][:, None] - e[neg][None, :]
        dh = h[pos][:, None] - h[neg][None, :]
        pairs += de.size
        opposite += int((np.sign(de) * np.sign(dh) < 0).sum())
        ties += int(((de == 0) | (dh == 0)).sum())
    print(f"  exhausted-terminated pairs within conditions: {pairs}; early entropy and the"
          f" step-0 baseline order {opposite} ({100*opposite/pairs:.1f}%) in opposite"
          f" directions, {ties} ties")
    y_all = at.ceiling.to_numpy().astype(int)
    v_all, arm_all, prob_all = at.entropy.to_numpy(), at.arm_id.to_numpy(), at.problem_id.to_numpy()
    a_c, a_p = stratified_auc(v_all, y_all, arm_all), stratified_auc(v_all, y_all, prob_all)
    lo_p, hi_p = cluster_ci(v_all, y_all, prob_all, prob_all, rng)
    print(f"  all {len(at)} trajectories at risk, no condition dropped: within condition {a_c:.3f},"
          f" within problem {a_p:.3f} [{lo_p:.3f}, {hi_p:.3f}]")
    # Temperature shifts the whole reading of a condition through the sampled tokens.
    # Replacing each value by its percentile within its own condition removes that shift
    # (and any other monotone one) before ranking within problem. It does not remove the
    # direct effect of temperature on the outcome, so this is a sensitivity check only.
    pct = sub.groupby("arm_id").entropy.rank(pct=True).to_numpy()
    a_r = stratified_auc(pct, y, prob)
    lo_r, hi_r = cluster_ci(pct, y, prob, prob, rng)
    print(f"  within problem, entropy as percentile within its condition: {a_r:.3f}"
          f" [{lo_r:.3f}, {hi_r:.3f}]")
    return {"aurocs": rows, "cond_minus_problem": {"d": d1, "lo": lo1, "hi": hi1},
            "entropy_minus_step0": {"d": d2, "lo": lo2, "hi": hi2},
            "pairs": {"n": pairs, "opposite": opposite, "ties": ties},
            "all_at_risk": {"n": len(at), "within_condition": a_c, "within_problem": a_p,
                            "lo": lo_p, "hi": hi_p},
            "percentile_within_condition": {"within_problem": a_r, "lo": lo_r, "hi": hi_r}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import yaml  # only needed for the entropy grid
    grid = yaml.safe_load((args.runs[0] / "config.yaml").read_text())["grid"]

    tr, st, report = load_corpus(args.runs)
    rng = np.random.default_rng(SEED)
    out = {"manifest": report}
    out["first_step"] = first_step_check(args.runs, tr)
    out["topk_recompute"] = topk_recompute_check(args.runs, grid, rng)
    out["difficulty"] = difficulty_coupling(tr, st)
    out["residual"] = residual_ranking(tr, st, rng)
    ev = onsets(tr, st)
    out["events"] = event_definitions(ev)
    out["detector"] = detector_sweep(ev, st)
    out["marker"] = marker_vs_detector(ev, st)
    out["details"] = text_details(tr, st)
    out["baselines"] = baselines(tr, st, rng)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
