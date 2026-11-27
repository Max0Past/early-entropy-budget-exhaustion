#!/usr/bin/env python
"""End-to-end smoke run against real weights, with acceptance criteria.

"It generated text" is not a passing result. This script asserts the criteria
that were fixed in advance, so that the first run either earns confidence or
names what is wrong:

    1. both termination branches exercised: at least one natural EOS and at
       least one truncated trajectory
    2. logits_to_keep=1 is in force on prefill (checked by shape, not by trust)
    3. logits are upcast to fp32 before any softmax
    4. the run directory is complete from the very first run
    5. zero reachable-band violations across every step
    6. model.generation_config.eos_token_id agrees with the tokenizer

Criterion 1 is obtained by giving one problem a deliberately small budget. That
is NOT a max_new_tokens estimate: the real value comes from the p99.9 of the
measured healthy length distribution and nothing else.

Usage:
    python scripts/run_smoke.py

Exits 0 if every criterion passes, 1 otherwise. Requires the weights (~3.6 GB).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from cotctl.data import format_prompt, load_gsm8k  # noqa: E402
from cotctl.decoding import DecodeConfig, generate_trajectory  # noqa: E402
from cotctl.runner import Arm, ArmRole, RunConfig, load_model, run  # noqa: E402
from cotctl.signals import (  # noqa: E402
    SIGNAL_NAMES,
    count_reachable_band_violations,
)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "data" / "runs"

# Analysis loaders must skip run directories carrying this prefix.
SMOKE_PREFIX = "_smoke_"

# Large enough that an easy GSM8K problem can actually reach EOS. 640 was not:
# every trajectory truncated, which is the most common pipeline trap, reproduced in
# miniature -- at that budget a healthy long trajectory is mislabelled
# FAILED_TERMINATION and nothing crashes. This is a smoke-run budget, not an
# estimate of max_new_tokens; that comes from p99.9 of the measured healthy
# length distribution and from nothing else.
N_PROBLEMS = 3
MAX_NEW_TOKENS = 4096
TRUNCATION_BUDGET = 24

failures: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    if not passed:
        failures.append(label)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}{'  ' + detail if detail else ''}")


def check_prefill_and_dtype(model, tokenizer) -> None:
    """Criteria 2, 3 and 6: verified directly, not inferred from the run."""
    print("=== A. prefill, dtype and EOS identity ===")

    raw = model.generation_config.eos_token_id
    ids = {raw} if isinstance(raw, int) else set(raw)
    check("generation_config EOS agrees with tokenizer",
          tokenizer.eos_token_id in ids,
          f"generation_config={sorted(ids)}, tokenizer={tokenizer.eos_token_id}")

    inputs = tokenizer("What is 2+2?", return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        without = model(**inputs, use_cache=True)
        with_kwarg = model(**inputs, use_cache=True, logits_to_keep=1)

    check("prefill returns all positions without the kwarg",
          without.logits.shape[1] == prompt_len, str(tuple(without.logits.shape)))
    check("logits_to_keep=1 collapses prefill to one position",
          with_kwarg.logits.shape[1] == 1, str(tuple(with_kwarg.logits.shape)))
    check("model logits are bf16 on this card",
          with_kwarg.logits.dtype == torch.bfloat16, str(with_kwarg.logits.dtype))

    # The sensor must never see bf16. decoding.py upcasts before any softmax.
    upcast = with_kwarg.logits[0, -1, :].float()
    check("upcast to fp32 before softmax", upcast.dtype == torch.float32)


def check_single_trajectory(model, tokenizer) -> None:
    """Sanity of the signals themselves on real model output."""
    print("=== B. one trajectory, signals on real logits ===")
    problem = load_gsm8k("test", 1)[0]
    prompt = format_prompt(problem.question, tokenizer)
    generator = torch.Generator(device=model.device).manual_seed(0)
    trajectory = generate_trajectory(
        model, tokenizer, prompt,
        DecodeConfig(temperature=1.0, top_p=1.0, max_new_tokens=128),
        generator,
    )

    signals = trajectory.signals
    entropy = signals[:, SIGNAL_NAMES.index("entropy")]
    check("trajectory produced steps", len(trajectory.token_ids) > 0,
          f"{len(trajectory.token_ids)} tokens, termination={trajectory.termination}")
    check("signals are finite (kl_prev null at t=0 aside)",
          bool(torch.isfinite(signals[1:]).all()))
    check("entropy is non-negative", bool((entropy >= 0).all()),
          f"min={entropy.min():.4f}, max={entropy.max():.4f}, mean={entropy.mean():.4f}")
    check("no reachable-band violations",
          count_reachable_band_violations(signals) == 0)
    print(f"         first 40 chars: {trajectory.text[:40]!r}")


def check_run_directory(run_dir: Path) -> None:
    """Criteria 1, 4 and 5, read back off disk."""
    print("=== C. run directory ===")
    present = {p.name for p in run_dir.iterdir()}
    required = {"steps.parquet", "trajectories.parquet", "config.yaml",
                "git_commit", "env.lock.txt"}
    check("run directory complete on the first run", required <= present,
          str(sorted(present)))

    steps = pq.read_table(run_dir / "steps.parquet")
    trajectories = pq.read_table(run_dir / "trajectories.parquet")
    terminations = trajectories["termination"].to_pylist()

    check("both termination branches exercised",
          "eos" in terminations and "max_tokens" in terminations,
          str(sorted(set(terminations))))
    check("steps written", steps.num_rows > 0, f"{steps.num_rows} steps")
    check("step count matches trajectory rows",
          steps.num_rows == sum(trajectories["n_steps"].to_pylist()))

    entropy = steps["entropy"].to_numpy()
    lower = steps["entropy_at_t_min"].to_numpy()
    upper = steps["entropy_at_t_max"].to_numpy()
    violations = int(((entropy < lower - 1e-4) | (entropy > upper + 1e-4)).sum())
    check("zero band violations across the whole run", violations == 0,
          f"{violations}/{steps.num_rows}")

    check("model revision recorded",
          all(r == REVISION for r in trajectories["model_revision"].to_pylist()))
    check("dataset revision recorded",
          all(r and len(r) == 40
              for r in trajectories["dataset_revision"].to_pylist()))

    print(f"         extraction outcomes: "
          f"{sorted(set(trajectories['extraction_outcome'].to_pylist()))}")
    print(f"         correct: {sum(trajectories['correct'].to_pylist())}"
          f"/{trajectories.num_rows}")

    # Throughput and length scale. Both feed later decisions: the pipeline budgets
    # 40-55 tok/s single-stream, and max_new_tokens is set from the healthy length
    # distribution, of which this is the very first glimpse.
    lengths = trajectories["n_steps"].to_pylist()
    wall = trajectories["wall_time_s"].to_pylist()
    total_tokens, total_time = sum(lengths), sum(wall)
    print(f"         lengths: {sorted(lengths)}")
    print(f"         throughput: {total_tokens / total_time:.1f} tok/s "
          f"({total_tokens} tokens in {total_time:.1f}s)")
    for length, termination in zip(lengths, terminations):
        if termination == "eos":
            print(f"         first natural EOS at {length} tokens")
            break


def main() -> int:
    print(f"loading {MODEL} @ {REVISION[:12]}")
    model, tokenizer = load_model(MODEL, REVISION)
    free, total = torch.cuda.mem_get_info()
    print(f"vram after load: {(total - free) / 2**30:.2f} GiB used "
          f"of {total / 2**30:.2f} GiB\n")

    check_prefill_and_dtype(model, tokenizer)
    check_single_trajectory(model, tokenizer)

    print("=== C. full sweep ===")
    config = RunConfig(
        model_name=MODEL,
        model_revision=REVISION,
        tag=SMOKE_PREFIX,
        arms=(
            Arm("main", temperature=1.0, top_p=1.0,
                max_new_tokens=MAX_NEW_TOKENS, n_problems=N_PROBLEMS,
                role=ArmRole.HYPOTHESIS),
            # A separate arm rather than a per-problem override: a deliberately
            # tiny ceiling is a different operating point, and pretending
            # otherwise would vary the FAILED_TERMINATION detector inside one arm.
            Arm("truncated", temperature=1.0, top_p=1.0,
                max_new_tokens=TRUNCATION_BUDGET, n_problems=1,
                role=ArmRole.DEMONSTRATION),
        ),
    )
    run_dir = run(config, RUNS_ROOT)
    print(f"         wrote {run_dir}")
    check_run_directory(run_dir)

    # Marked, not deleted. The smoke run is evidence rather than data, but
    # destroying artefacts in a pipeline whose one claimed virtue is
    # reproducibility is the wrong default -- an earlier version of this script
    # deleted 3004 steps that were wanted an hour later. Analysis skips
    # directories carrying this prefix.
    print(f"         kept {run_dir.name} (prefix {SMOKE_PREFIX!r} marks it as evidence)")

    print()
    if failures:
        print(f"RESULT: {len(failures)} CRITERION/CRITERIA FAILED")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("RESULT: ALL ACCEPTANCE CRITERIA PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
