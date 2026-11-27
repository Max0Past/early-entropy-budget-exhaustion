#!/usr/bin/env python
"""The paper's figures, drawn from the corpus and from the analysis scripts' JSON.

Summary numbers (rates, AUROCs, the detector sweep) are read from the JSON that
`analyze_corpus.py` and `paper_numbers.py` write, so a figure cannot disagree with
the printed tables. Per-trajectory and per-problem values are recomputed from the
runs with the same functions, and checked against the JSON where both exist.

  Fig. 1  instrument check and failure to terminate, both against sampling temperature
  Fig. 2  entropy change at the ceiling by arm; early-warning AUROC by arm vs by problem
  Fig. 3  the persistence detector on the 47 exactly periodic failures, under two onset
          definitions

Usage:
    python scripts/analyze_corpus.py RUN_DIR [RUN_DIR ...] --json results/analyze_corpus.json
    python scripts/paper_numbers.py  RUN_DIR [RUN_DIR ...] --json results/paper_numbers.json
    python scripts/make_figures.py   RUN_DIR [RUN_DIR ...] --results results --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_corpus import load_corpus  # noqa: E402
from paper_numbers import (  # noqa: E402
    MARKER_PRIMARY, entropy_from_logits, first_fire, new_ngram_flags, onsets,
    read_steps, repetition_marker,
)

WIDTH: Final[float] = 4.8          # inches; the text width of an LNCS page
HEIGHT: Final[float] = 2.1
BLACK: Final[str] = "black"
GREY: Final[str] = "0.55"
# The setting quoted in the text for the 47 periodic failures: the most fires before
# the exact onset at no more than 5 % false alarms.
EXAMPLE_SETTING: Final[tuple[float, int]] = (0.5, 400)
TAIL_BASELINE: Final[int] = 256    # same windows as analyze_corpus.py, section 5
TAIL_WINDOW: Final[int] = 1000
TAIL_MIN_LENGTH: Final[int] = 1400

STYLE: Final[dict] = {
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.formatter.use_locale": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "savefig.dpi": 300, "pdf.fonttype": 42,
}


def panel_label(ax, letter: str) -> None:
    ax.text(-0.2, 1.06, letter, transform=ax.transAxes, fontweight="bold", va="bottom")


def arm_label(arm: str, temperature: float) -> str:
    """'0.3 cold', '0.6 reference', ... for the rows of Fig. 2a."""
    role = arm.split("-")[0].replace("hyp", "hot")
    return f"{temperature:.1f} {role}"


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.pdf and .png")


# --- Fig. 1 -------------------------------------------------------------------------

def first_step_by_temperature(runs: list[Path], tr: pd.DataFrame) -> pd.DataFrame:
    """Step-0 entropy, invariant and tempered, on the problems attempted at every T.

    The two T = 0.3 arms cover complementary problem sets after de-duplication, so
    they are pooled into one condition. Restricting to common problems keeps arm
    means from differing through problem subsets, which they otherwise do.
    """
    s0 = read_steps(runs, ["traj_id", "step", "entropy", "topk_logits"],
                    ds.field("step") == 0)
    s0 = s0.merge(tr[["run", "traj_id", "problem_id", "temperature"]],
                  on=["run", "traj_id"])
    logits = np.stack(s0.topk_logits.to_numpy())
    s0["tempered"] = [float(entropy_from_logits(z, t))
                      for z, t in zip(logits, s0.temperature)]
    s0["T"] = s0.temperature.round(3)
    common = set.intersection(*(set(g.problem_id) for _, g in s0.groupby("T")))
    sub = s0[s0.problem_id.isin(common)]
    return (sub.groupby("T")
               .agg(n=("problem_id", "nunique"),
                    inv=("entropy", "mean"), inv_sd=("entropy", "std"),
                    tmp=("tempered", "mean"), tmp_sd=("tempered", "std"))
               .reset_index())


def figure_1(runs: list[Path], tr: pd.DataFrame, corpus: dict, out: Path) -> None:
    fs = first_step_by_temperature(runs, tr)
    fig, (a, c) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))

    a.errorbar(fs["T"], fs.inv, yerr=fs.inv_sd, color=BLACK, marker="o", ms=3.5,
               capsize=2, label="invariant ($T_{ref}=1$)")
    a.errorbar(fs["T"], fs.tmp, yerr=fs.tmp_sd, color=GREY, marker="s", ms=3.5,
               mfc="white", ls="--", capsize=2, label="tempered (arm's own $T$)")
    a.set_xlabel("sampling temperature of the arm")
    a.set_ylabel("entropy at step 0 (nats)")
    a.set_ylim(-0.1, 1.25)
    a.legend(loc="upper left")
    panel_label(a, "a")

    rows = pd.DataFrame(corpus["ceiling_by_arm"])
    offset = {"cold-T0.3": -0.012, "demo-T0.3": 0.012}
    for r in rows.itertuples():
        x = r.T + offset.get(r.arm, 0.0)
        hollow = r.top_p < 1.0
        c.errorbar(x, r.pct, yerr=[[r.pct - r.lo], [r.hi - r.pct]], color=BLACK,
                   marker="o", ms=3.5, mfc="white" if hollow else BLACK, capsize=2)
    ref = rows[rows.top_p < 1.0].iloc[0]
    c.annotate("top-p 0.95", (ref["T"], ref.pct), xytext=(8, 10),
               textcoords="offset points", fontsize=6.5, color=GREY)
    c.set_xlabel("sampling temperature")
    c.set_ylabel("exhausted the budget (%)")
    c.set_ylim(0, 35)
    panel_label(c, "b")

    fig.tight_layout(w_pad=2.5)
    save(fig, out, "fig1_sensor_and_failure_rate")


# --- Fig. 2 -------------------------------------------------------------------------

def tail_entropy_change(tr: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Per ceiling trajectory: median of the last 1000 steps minus median of the first 256."""
    ent = {k: g.entropy.to_numpy() for k, g in st.groupby(["run", "traj_id"], sort=False)}
    rows = []
    for r in tr[tr.ceiling].itertuples():
        h = ent[(r.run, r.traj_id)]
        if len(h) < TAIL_MIN_LENGTH:
            continue
        rows.append(dict(arm=r.arm_id, T=float(r.temperature),
                         dH=float(np.median(h[-TAIL_WINDOW:]) - np.median(h[:TAIL_BASELINE]))))
    return pd.DataFrame(rows)


def figure_2(tr: pd.DataFrame, st: pd.DataFrame, corpus: dict, numbers: dict,
             out: Path) -> None:
    dd = tail_entropy_change(tr, st)
    ref = {r["arm"]: r["dH_median"] for r in corpus["tail_entropy"]}
    for arm, g in dd.groupby("arm"):
        if not np.isclose(g.dH.median(), ref[arm]):
            raise SystemExit(f"Fig. 2a disagrees with analyze_corpus.py for {arm}")

    fig, (d, b) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT + 0.2),
                               gridspec_kw={"width_ratios": [1.1, 1]})
    order = (dd.groupby(["arm", "T"]).size().reset_index()
               .sort_values(["T", "arm"], ascending=[True, False]))
    rng = np.random.default_rng(0)   # vertical jitter only
    for y, r in enumerate(order.itertuples()):
        v = dd[dd.arm == r.arm].dH.to_numpy()
        d.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=5, color=GREY, lw=0)
        m = np.median(v)
        d.plot([m, m], [y - 0.32, y + 0.32], color=BLACK, lw=1.6)
    d.axvline(0, color=GREY, lw=0.6, ls=":")
    d.set_yticks(range(len(order)))
    d.set_yticklabels([f"{arm_label(r.arm, r.T)} (n={r[3]})" for r in order.itertuples()])
    d.set_xlabel("entropy change, tail vs start (nats)")
    panel_label(d, "a")

    auc = pd.DataFrame(corpus["auc"])
    auc = auc[auc.signal == "entropy"].sort_values("K")
    x = np.arange(len(auc))
    b.errorbar(x - 0.1, auc.by_arm, yerr=[auc.by_arm - auc.by_arm_lo, auc.by_arm_hi - auc.by_arm],
               color=BLACK, marker="o", ms=3.5, ls="none", capsize=2, label="within condition")
    b.errorbar(x + 0.1, auc.by_problem,
               yerr=[auc.by_problem - auc.by_problem_lo, auc.by_problem_hi - auc.by_problem],
               color=GREY, marker="s", ms=3.5, mfc="white", ls="none", capsize=2,
               label="within problem")
    b.axhline(0.5, color=GREY, lw=0.6, ls=":")
    # The prompt-only baseline (entropy at step 0, within condition) sees no trajectory,
    # so it is one value for every K; it is drawn only at K = 256, where it was computed.
    base = numbers["baselines"]["aurocs"]["minus entropy at step 0"]["within_condition"]
    b.plot([-0.35, 0.35], [base, base], color=BLACK, lw=1.0, ls="--",
           label="prompt only")
    b.set_xticks(x)
    b.set_xticklabels([str(k) for k in auc.K])
    b.set_xlabel("prefix length K (steps)")
    b.set_ylabel("AUROC of early entropy")
    b.set_ylim(0.25, 1.0)
    b.legend(loc="upper right")
    panel_label(b, "b")

    fig.tight_layout(w_pad=2.0)
    save(fig, out, "fig2_ceiling_regimes_and_confound")


# --- Fig. 3 -------------------------------------------------------------------------

def frontier(fa_pct: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Most trajectories caught before onset at or below each false-alarm rate."""
    order = np.argsort(fa_pct, kind="stable")
    return fa_pct[order], np.maximum.accumulate(count[order])


def periodic_timelines(tr: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Marker step, exact onset and detector firing step for every periodic failure."""
    ev = onsets(tr, st)
    per = ev[ev.per_onset.notna()]
    window, level = MARKER_PRIMARY
    groups = {k: g for k, g in st.groupby(["run", "traj_id"], sort=False)}
    rows = []
    for r in per.itertuples():
        g = groups[(r.run, r.traj_id)]
        rows.append(dict(marker=repetition_marker(new_ngram_flags(g.token_id.to_numpy()),
                                                  window, level),
                         exact=int(r.per_onset),
                         fire=first_fire(g.entropy.to_numpy(), *EXAMPLE_SETTING)))
    return pd.DataFrame(rows)


def figure_3(tr: pd.DataFrame, st: pd.DataFrame, numbers: dict, out: Path) -> None:
    sweep = pd.DataFrame(numbers["detector"]["sweep"])
    sweep = sweep[sweep.target == "periodic"]
    n_neg = int(sweep.n_negative.iloc[0])
    n_t = int(sweep.n_targets.iloc[0])
    window, level = MARKER_PRIMARY
    mk = next(m for m in numbers["marker"]["markers"]
              if m["window"] == window and np.isclose(m["level"], level))
    ms = pd.DataFrame(mk["settings"])

    tl = periodic_timelines(tr, st)
    quoted = numbers["marker"]["level_0.5_pers_400"]
    early = tl[tl.fire < tl.exact]
    if len(early) != quoted["before_exact"] or (early.fire >= early.marker).sum() != quoted["after_marker"]:
        raise SystemExit("Fig. 3b disagrees with paper_numbers.py, section 13")

    fig, (e, f) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT + 0.3))
    for data, col, label, style in (
            (sweep.rename(columns={"before_onset": "n"}), GREY, "onset = exact periodicity",
             dict(marker="s", mfc="white")),
            (ms.rename(columns={"before_marker": "n"}), BLACK, "onset = repetition marker",
             dict(marker="o", mfc=BLACK))):
        x = 100 * data.false_alarms.to_numpy() / n_neg
        y = data.n.to_numpy()
        e.plot(x, y, ls="none", color=col, ms=3, alpha=0.8, **style)
        fx, fy = frontier(x, y)
        e.step(fx, fy, where="post", color=col, lw=0.9, label=label)
    e.set_xscale("symlog", linthresh=1, linscale=0.5)
    e.set_xlim(-0.1, 110)
    e.set_xticks([0, 1, 5, 20, 100])
    e.set_xticklabels(["0", "1", "5", "20", "100"])
    e.set_xlabel("false alarms (% of terminated trajectories)")
    e.set_ylabel(f"fires before onset (of {n_t})")
    e.legend(loc="upper left")
    panel_label(e, "a")

    tl = tl.sort_values("marker", ascending=False).reset_index(drop=True)
    y = np.arange(len(tl))
    f.hlines(y, tl.marker, tl.exact, color=GREY, lw=0.6)
    f.plot(tl.marker, y, "o", color=BLACK, ms=2.2, label="repetition marker")
    f.plot(tl.exact, y, "s", color=GREY, mfc="white", ms=2.2, mew=0.6, label="exact onset")
    f.plot(tl.fire, y, "x", color=BLACK, ms=2.6, mew=0.7, label="detector fires")
    f.set_xlim(0, 8192)
    f.set_yticks([])
    f.set_ylabel(f"{len(tl)} periodic failures")
    f.set_xlabel("decoding step")
    panel_label(f, "b")

    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.92))
    fig.legend(*f.get_legend_handles_labels(), loc="upper right", ncol=3,
               bbox_to_anchor=(1.0, 1.0), columnspacing=0.8, handletextpad=0.2)
    save(fig, out, "fig3_detector_vs_onset")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--results", type=Path, required=True,
                    help="directory with analyze_corpus.json and paper_numbers.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    corpus = json.loads((args.results / "analyze_corpus.json").read_text(encoding="utf-8"))
    numbers = json.loads((args.results / "paper_numbers.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    tr, st, _ = load_corpus(args.runs)
    figure_1(args.runs, tr, corpus, args.out)
    figure_2(tr, st, corpus, numbers, args.out)
    figure_3(tr, st, numbers, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
