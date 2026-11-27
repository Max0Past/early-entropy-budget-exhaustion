"""Trajectory classes from outcome x termination, and `t_onset` from surface markers.

The separation between those two things is the whole point of this module, and it
is the thing a reviewer will attack first.

**Classes come from ground truth and from the generation loop. Never from the
signal being tested, and never from the text surface.** A class defined by
repetition would make H1a circular: repetition *means* a peaked, low-entropy
distribution by construction, so "discovering" that such trajectories have low
entropy proves nothing.

**Surface markers locate `t_onset` and nothing else.** `t_onset` exists because
H1b measures lead time *relative* to failure onset. It never decides a class.

The `COLLAPSE` / `DRIFT` split is **descriptive stratification only**. It is
computed here because the two regimes are worth characterising, and it must be
stated in writing that it is not used to test H1a. See the Labelling section of the README.

Thresholds in this module are **provisional**. They must be calibrated against the
healthy reference arm before any of them is used on results: a marker threshold
chosen by eye on failure trajectories is a threshold chosen to find what you
expected.
"""

from __future__ import annotations

import zlib
from enum import StrEnum
from typing import Final

# --- provisional, calibrate on the healthy arm before use --------------------
# A window long enough to contain a repetition cycle but short enough to locate
# onset to within a useful number of steps.
ONSET_WINDOW: Final[int] = 128
ONSET_STRIDE: Final[int] = 16
# Fraction of n-grams in a window that are repeats. Healthy prose repeats a fair
# amount, so this is deliberately well above zero.
REPETITION_THRESHOLD: Final[float] = 0.65
# Windows the marker must stay tripped for, so a single repeated phrase does not
# register as onset.
ONSET_PERSISTENCE: Final[int] = 3
REPETITION_NGRAM: Final[int] = 8
# Tail of the generation examined for the descriptive COLLAPSE / DRIFT split.
STRATIFY_TAIL: Final[int] = 512


class TrajectoryClass(StrEnum):
    """The label. Derived only from termination, correctness and extractability."""

    HEALTHY = "healthy"
    SEMANTIC_ERROR = "semantic_error"
    FAILED_TERMINATION = "failed_termination"
    DEGENERATE_LUCKY = "degenerate_lucky"
    EXTRACTION_FAILED = "extraction_failed"


class FailureStratum(StrEnum):
    """Descriptive stratification of `FAILED_TERMINATION`. Not a label.

    Never use this to test H1a. It exists to characterise the two regimes and to
    report them separately, and the thesis must say so explicitly.
    """

    COLLAPSE = "collapse"
    DRIFT = "drift"
    UNCLASSIFIED = "unclassified"


# Outcomes on which no reliable correctness judgement is possible. Both are
# counted and excluded rather than scored. `span_found_unparseable` is the
# subtler one: extraction "worked", but the span is not a number, so correctness
# would fall back to string comparison and almost always fail -- landing the
# trajectory in SEMANTIC_ERROR for a formatting reason. GSM8K gold is always an
# integer, so a non-numeric answer cannot be scored either way.
UNSCORABLE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"no_span", "span_found_unparseable"}
)


def classify(
    termination: str, correct: bool, extraction_outcome: str
) -> TrajectoryClass:
    """Assign the trajectory class.

    Args:
        termination: "eos" or "max_tokens", straight from the decode loop.
        correct: whether the extracted answer matched gold.
        extraction_outcome: "parsed_decimal", "span_found_unparseable" or
            "no_span".

    Note the asymmetry between the two columns, which is deliberate:

    * On **EOS**, an unscorable extraction is `EXTRACTION_FAILED`, not
      `SEMANTIC_ERROR`. We cannot say what answer the model gave, so calling it a
      semantic error would manufacture the negative control that decides H1c.
      Counted and excluded, exactly like `DEGENERATE_LUCKY`. The two unscorable
      outcomes stay distinguishable in the data even though they share a class,
      so the audit can quantify each without re-reading generations.
    * On **truncation**, the label comes from termination rather than from answer
      content, so a failed extraction changes nothing and the trajectory stays
      `FAILED_TERMINATION`.
    """
    if termination not in ("eos", "max_tokens"):
        raise ValueError(f"unknown termination: {termination!r}")

    if termination == "max_tokens":
        return (
            TrajectoryClass.DEGENERATE_LUCKY if correct
            else TrajectoryClass.FAILED_TERMINATION
        )

    if extraction_outcome in UNSCORABLE_OUTCOMES:
        return TrajectoryClass.EXTRACTION_FAILED
    return TrajectoryClass.HEALTHY if correct else TrajectoryClass.SEMANTIC_ERROR


def is_analysed(label: TrajectoryClass) -> bool:
    """Whether the class enters the H1a comparison.

    `DEGENERATE_LUCKY` and `EXTRACTION_FAILED` are both counted and reported, and
    both excluded. Silently dropping either would hide exactly the contamination
    the counts exist to expose.
    """
    return label not in (
        TrajectoryClass.DEGENERATE_LUCKY,
        TrajectoryClass.EXTRACTION_FAILED,
    )


def repetition_rate(token_ids: list[int], n: int = REPETITION_NGRAM) -> float:
    """Fraction of n-grams in the sequence that are not unique.

    0.0 means every n-gram occurs once; values approaching 1.0 mean the sequence
    is dominated by a repeating cycle. Computed on token ids rather than text so
    that tokenizer artefacts do not enter the measurement.
    """
    if len(token_ids) < n + 1:
        return 0.0

    ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def lz_ratio(text: str) -> float:
    """Compressed length over raw length. Low means repetitive, high means random.

    Only meaningful *relative* to the same statistic measured on healthy text: the
    absolute value depends on the encoding and carries no interpretation on its
    own. This is why the split below is calibrated rather than thresholded at a
    number pulled from the air.
    """
    if not text:
        return 1.0

    raw = text.encode("utf-8")
    return len(zlib.compress(raw, level=6)) / len(raw)


def find_onset(
    token_ids: list[int],
    window: int = ONSET_WINDOW,
    stride: int = ONSET_STRIDE,
    threshold: float = REPETITION_THRESHOLD,
    persistence: int = ONSET_PERSISTENCE,
) -> int | None:
    """Locate the step at which repetition takes over, or None.

    Returns the first index of the first window in a run of `persistence`
    consecutive windows whose repetition rate exceeds `threshold`. The
    persistence requirement is what stops one repeated phrase from registering as
    onset; it is the same idea as the controller's persistence threshold, and for
    the same reason.

    **This locates `t_onset` for H1b. It does not define a class.** A trajectory
    with no detectable onset returns None, which is information, not a failure.
    """
    if len(token_ids) < window:
        return None

    consecutive = 0
    for start in range(0, len(token_ids) - window + 1, stride):
        if repetition_rate(token_ids[start:start + window]) > threshold:
            consecutive += 1
            if consecutive == persistence:
                return start - (persistence - 1) * stride
        else:
            consecutive = 0
    return None


def stratify_failure(
    text: str,
    healthy_lz_low: float,
    healthy_lz_high: float,
    tail: int = STRATIFY_TAIL,
) -> FailureStratum:
    """Split `FAILED_TERMINATION` into collapse-like and drift-like, descriptively.

    Args:
        text: the generated text.
        healthy_lz_low: lower edge of the healthy `lz_ratio` band.
        healthy_lz_high: upper edge of the healthy `lz_ratio` band.
        tail: how many characters of the end of the generation to examine.

    The two bounds are **required arguments with no defaults**, because they can
    only come from the healthy reference arm. Hard-coding them would be choosing a
    threshold that finds what was expected.

    Returns `UNCLASSIFIED` when the tail sits inside the healthy band — a
    trajectory that failed to terminate without either signature. That is a real
    outcome and must be reported rather than forced into one of the two buckets.

    **Never use the result to test H1a.**
    """
    ratio = lz_ratio(text[-tail:])
    if ratio < healthy_lz_low:
        return FailureStratum.COLLAPSE
    if ratio > healthy_lz_high:
        return FailureStratum.DRIFT
    return FailureStratum.UNCLASSIFIED


# --- exact periodicity: the blind spot of the n-gram marker above -------------
# `find_onset` cannot see a cycle longer than ~42 tokens, and the arithmetic is
# forced, not incidental. A window of ONSET_WINDOW = 128 contains
# 128 - REPETITION_NGRAM + 1 = 121 n-gram positions, and a rate above
# REPETITION_THRESHOLD = 0.65 requires unique/total < 0.35, i.e. at most 42
# distinct 8-grams. A cycle of period p contributes on the order of p distinct
# 8-grams, so any cycle with p > 42 is invisible to that marker no matter how
# long it persists. Measured on this corpus the median period is 148, so the
# marker finds 6 of the 47 trajectories that are exactly periodic.
#
# The two functions below test periodicity directly and have no upper bound on
# period other than the one passed in. They are descriptive, like the markers
# above, and carry the same prohibition: they locate a regime, they never
# define a class. See the Labelling section of the README.
PERIODIC_TAIL: Final[int] = 1024
PERIODIC_MAX_PERIOD: Final[int] = 512


def tail_period(
    token_ids: list[int],
    tail: int = PERIODIC_TAIL,
    max_period: int = PERIODIC_MAX_PERIOD,
) -> int | None:
    """Smallest `p` for which the last `tail` tokens repeat with period `p`.

    Returns `None` when no period at or below `max_period` fits, which includes
    the case of a sequence shorter than `tail`.

    Exactness is the point. A cycle that drifts by one token is not detected,
    and that is intended: this is a test for the degenerate attractor, not a
    similarity measure. Approximate repetition is what `repetition_rate` and
    `lz_ratio` are for.
    """
    if len(token_ids) < tail:
        return None

    a = token_ids[-tail:]
    for p in range(1, min(max_period, tail - 1) + 1):
        if a[p:] == a[:-p]:
            return p
    return None


def periodicity_onset(token_ids: list[int], period: int) -> int:
    """Earliest step from which the sequence stays `period`-periodic to the end.

    Binary search over the suffix start. The predicate is monotone: if the suffix
    from `t` is periodic then so is the suffix from any `t' > t`, because the
    latter is a suffix of the former. That monotonicity is what makes the search
    valid, and it fails if `period` is not an actual period of the tail — so pass
    the value `tail_period` returned, not a guess.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    lo, hi = 0, max(0, len(token_ids) - period)
    while lo < hi:
        mid = (lo + hi) // 2
        if token_ids[mid + period:] == token_ids[mid:-period]:
            hi = mid
        else:
            lo = mid + 1
    return lo
