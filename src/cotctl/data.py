"""Dataset loading, prompt formatting, and answer extraction.

Only GSM8K is implemented. MATH is named in the pipeline but adding a second
extractor before the first one has been hand-audited would just double the
surface area of the failure below.

**Answer extraction is the highest-risk code in Phase I**, and not for the usual
reasons. Every extractor failure is scored as a wrong answer, and a wrong answer
that terminated on EOS is labelled `SEMANTIC_ERROR` -- the negative control that
decides H1c. A sloppy extractor therefore does not merely add noise: it
manufactures the exact class whose emptiness the thesis depends on.

Two consequences, both load-bearing:

* Failure to extract returns None and is reported distinctly from an
  extracted-but-wrong answer, so contamination of the negative control is
  measurable rather than invisible.
* The permissive fallback (take the last number in the text) is OFF by default.
  It converts "could not parse" into a confident guess, which is precisely the
  wrong trade here.

The study protocol requires ~50 extractions to be audited by hand before any number
from this module is trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final, Literal

from datasets import load_dataset

GSM8K_REPO: Final[str] = "openai/gsm8k"
GSM8K_CONFIG: Final[str] = "main"

# Pinned, not merely recorded. Recording the SHA lets you notice a divergence
# after the fact; pinning stops it happening. Invariant 8 applies to data with
# the same force as it applies to weights.
GSM8K_REVISION: Final[str] = "740312add88f781978c0658806c59bc2815b9866"

# GSM8K puts the gold answer after this marker, on the last line.
GOLD_MARKER: Final[str] = "####"

# From the DeepSeek-R1 model card: no system prompt, instructions in the user
# turn, and an explicit \boxed{} request for maths. The \boxed{} request is what
# makes extraction reliable -- without it the model ends in free prose and the
# extractor has to guess.
MATH_INSTRUCTION: Final[str] = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Prefilled onto the assistant turn so that step 0 is the same regime for every
# trajectory. See `format_prompt`.
THINK_OPEN: Final[str] = "<think>\n"

# Three mutually exclusive outcomes, not two. The middle one is the class that
# would otherwise hide: a span WAS found, so extraction "worked", but it does not
# parse as a number, so correctness falls back to string comparison and almost
# always fails -- landing a trajectory in SEMANTIC_ERROR for a formatting reason.
# Splitting it out makes that contamination visible in Parquet without any manual
# reading, because Decimal already raises on exactly this case.
ExtractionOutcome = Literal["parsed_decimal", "span_found_unparseable", "no_span"]


@dataclass(frozen=True, slots=True)
class Problem:
    """One problem, independent of any model or prompt format."""

    problem_id: str
    question: str
    gold_answer: str


@dataclass(frozen=True, slots=True)
class ExtractedAnswer:
    """A parsed answer, the outcome of parsing it, and the span it came from.

    `outcome` is persisted so the negative control can be audited mechanically
    rather than by eye. `span_found_unparseable` in particular is the failure a
    hand-audit is bad at finding and an exception type is good at finding.

    `raw` is the matched span before normalisation. It costs a few bytes per
    trajectory and it is what makes the mandatory hand-audit a matter of reading
    Parquet instead of regenerating the run.
    """

    value: str | None
    outcome: ExtractionOutcome
    raw: str | None = None


def resolve_dataset_revision(repo_id: str = GSM8K_REPO) -> str | None:
    """Current commit SHA of the dataset repo, for the reproducibility record."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo_id).sha
    except Exception:
        return None


def extract_gold_answer(raw_answer: str) -> str:
    """Pull the gold answer out of a GSM8K solution.

    The solution is worked prose followed by `#### <answer>`. Raises rather than
    returning a sentinel: a missing marker means the dataset is not what we think
    it is, and silently continuing would corrupt every correctness label.
    """
    if GOLD_MARKER not in raw_answer:
        raise ValueError(f"no {GOLD_MARKER!r} marker in GSM8K answer: {raw_answer!r}")
    return normalise_number(raw_answer.rsplit(GOLD_MARKER, 1)[1])


def load_gsm8k(
    split: str = "test",
    limit: int | None = None,
    revision: str = GSM8K_REVISION,
) -> list[Problem]:
    """Load GSM8K as plain `Problem` records.

    Args:
        split: "train" or "test".
        limit: take only the first n problems, for smoke runs.
        revision: dataset commit SHA. Defaults to the pin; overriding it makes
            the resulting run incomparable with every other run.
    """
    dataset = load_dataset(GSM8K_REPO, GSM8K_CONFIG, split=split, revision=revision)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [
        Problem(
            problem_id=f"gsm8k-{split}-{index}",
            question=row["question"],
            gold_answer=extract_gold_answer(row["answer"]),
        )
        for index, row in enumerate(dataset)
    ]


def format_prompt(question: str, tokenizer, force_think: bool = True) -> str:
    """Render one problem through the model's chat template.

    The template, not a hand-built string: R1-distill opens the assistant turn
    with its own thinking delimiter, and reproducing that by hand is exactly the
    kind of detail that drifts between model families. No system prompt -- the
    R1 model card puts every instruction in the user turn, which is why
    MATH_INSTRUCTION lives there and why the share of `boxed` extractions
    depends on it.

    `force_think` guarantees the assistant turn opens with the think tag. This is
    a measurement decision, not a quality one. The R1 series sometimes skips the
    reasoning mode and emits an empty `<think>\\n\\n</think>`, so without the
    prefill a fraction of trajectories would begin with the model deciding
    whether to think at all. That makes t = 0 mean two different things across
    the corpus, and a tau_burst averaged over both regimes measures nothing.

    Verified against R1-Distill-Qwen-1.5B at the pinned revision: its template
    already emits `<think>\\n` under `add_generation_prompt`, so here this is a
    guard that correctly does nothing rather than an edit that double-inserts.
    It still matters for the control model, whose template handles thinking
    differently.
    """
    messages = [{"role": "user", "content": f"{question}\n\n{MATH_INSTRUCTION}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if force_think and not prompt.rstrip().endswith(THINK_OPEN.strip()):
        prompt += THINK_OPEN
    return prompt


def normalise_number(text: str) -> str:
    """Strip presentation so that "$1,000." and "1000" compare equal.

    Deliberately conservative: it removes formatting, never rewrites the value.
    Aggressive normalisation is how wrong answers start scoring as correct.
    """
    cleaned = text.strip()
    # Escaped forms first: stripping "$" before "\$" would leave a stray
    # backslash and turn a correct "\$1,000" into "\1000", i.e. score a correct
    # answer as wrong and push it into the SEMANTIC_ERROR negative control.
    cleaned = cleaned.replace("\\$", "").replace("$", "")
    cleaned = cleaned.replace("\\%", "").replace("%", "")
    cleaned = cleaned.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.rstrip(".")
    return cleaned


def _extract_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...}, matching braces so nesting survives."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    cursor = start + len(marker)
    depth = 1
    for index in range(cursor, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[cursor:index]
    return None  # unbalanced braces, e.g. the generation was truncated mid-answer


def extract_pred_answer(text: str, allow_fallback: bool = False) -> ExtractedAnswer:
    """Extract the model's final answer.

    Args:
        text: the decoded generation.
        allow_fallback: if True, fall back to the last number in the text when no
            \\boxed{} is present. Off by default -- see the module docstring. A
            truncated collapse trajectory has no \\boxed{} and the fallback would
            hand it a confident wrong answer, moving it out of the class it
            belongs in.
    """
    span = _extract_boxed(text)
    if span is None and allow_fallback:
        numbers = re.findall(r"-?\d+(?:[.,]\d+)*", text)
        span = numbers[-1] if numbers else None

    if span is None:
        return ExtractedAnswer(None, "no_span", raw=None)

    value = normalise_number(span)
    # Decimal already distinguishes the two failure modes for us. A span that
    # normalises to something non-numeric ("eighteen", "\frac{1}{2}") is not a
    # semantic error -- it is an answer we cannot score -- and must not be
    # allowed to reach the negative control by way of a failed string compare.
    outcome = "parsed_decimal" if _to_decimal(value) is not None else "span_found_unparseable"
    return ExtractedAnswer(value, outcome, raw=span)


def _to_decimal(text: str) -> Decimal | None:
    """Parse exactly, or None. Decimal rather than float: no binary rounding, so
    equality is exact and no epsilon has to be invented."""
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def answers_match(pred: str | None, gold: str) -> bool:
    """Compare a predicted answer against gold.

    Numeric comparison first, so "18.0" matches "18" and "1000" matches "1000."
    without any of that depending on string normalisation having anticipated the
    exact formatting. String comparison is only the fallback for answers that are
    not numbers at all.

    This ordering is deliberate. GSM8K gold is always an integer, so anything
    that fails to parse numerically is already suspect, and every normalisation
    rule that fails to anticipate a LaTeX form silently pushes a correct answer
    into SEMANTIC_ERROR -- the negative control for H1c. Normalisation is an
    endless list of exceptions; numeric equality is not.
    """
    if pred is None:
        return False

    pred_value, gold_value = _to_decimal(pred), _to_decimal(gold)
    if pred_value is not None and gold_value is not None:
        return pred_value == gold_value
    return pred == gold
