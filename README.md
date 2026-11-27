# Does early entropy warn of token-budget exhaustion in chain-of-thought decoding?

Code, analysis and figures for the preprint of the same title (Pastushenko, 2026).
Preprint: [10.5281/zenodo.24605925](https://doi.org/10.5281/zenodo.24605925). Trajectories: [10.5281/zenodo.24606040](https://doi.org/10.5281/zenodo.24606040).

## The question

A reasoning model sometimes never emits an end-of-sequence token and decodes until the
token budget stops it. Adaptive decoders read a per-step statistic, usually entropy, and
change decoding in response, motivated in part by reports that early entropy anticipates
such failures. This repository asks whether it does, on two time scales: early in the
trajectory, and close to failure.

## Main result

On this corpus entropy shows nothing that is not already visible elsewhere: early in the
trajectory it is not shown to add to the prompt, and close to failure it trails the text.
Numbers are from `reproduce.sh` on the released trajectories.

- **Early in the trajectory: no more than the prompt.** Within a decoding condition,
  mean entropy over steps 128–255 ranks the trajectories that will exhaust the budget
  with AUROC 0.742 [0.672, 0.804]. A prompt-only score, the negative entropy of the
  first step (sign chosen after inspecting the data), gives 0.721; the paired difference
  is 0.022 [−0.069, 0.126], and the two scores order a third of the pairs differently.
  Within a problem, early-window entropy gives 0.441 [0.355, 0.527], and 0.500
  [0.403, 0.596] after each value is replaced by its percentile within its condition.
  In this design attempts at one problem always differ in temperature, so that comparison
  cannot isolate information about the individual attempt; repeated seeds at fixed
  temperature would.
- **Two regimes of exhaustion.** At T ≤ 0.3 the text of exhausted trajectories repeats
  itself (47 of 68 end in an exactly periodic tail) and entropy falls; at T = 1.0 it
  rises (6 of 6) and the text keeps changing.
- **Close to failure: the alarm follows the loop.** On the 47 loops with an exactly
  periodic tail, a persistence detector on entropy at level 0.5 nats and persistence
  400 steps (3.8 % false alarms) fires before the exact-periodicity onset on 18, and all
  18 come after a token-level repetition marker. No detector setting with at most 10 %
  false alarms fires before that marker, under any of six variants of it. As a stop rule
  the detector is useful: at level 0.05 nats and persistence 400 steps it stops 19 of 84
  exhausted trajectories with no false alarm on the 636 terminated ones.

The comparisons pool decoding temperatures, so entropy is read from the raw logits at a
fixed reference temperature (T_ref = 1). At the first decoding step this reading is
identical across conditions on every problem, whereas the entropy of the distribution
actually sampled from gives away the sampling temperature (pairwise AUROC 0.755–1.000,
mean 0.963 over 15 pairs of temperatures).

Intervals are 95 % cluster bootstrap with problems as the resampling unit; a problem
drawn more than once enters as separate strata.

## Setup

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` at revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`, bf16, no quantisation |
| Data | GSM8K `main`, test split, first 124 problems, revision `740312add88f781978c0658806c59bc2815b9866` |
| Decoding | own decode loop (`src/cotctl/decoding.py`), not `model.generate`; repetition penalties off; at most 8192 new tokens; batch 4; one seed per problem and arm |
| Sensor | entropy of softmax(z / T_ref) over the raw logits z, T_ref = 1, so the reading does not depend on the sampling temperature |
| Also stored per step | margin, surprisal, KL to the previous step, logit variance, entropy at 7 temperatures, top-32 token ids and logits |
| Corpus | 772 trajectories in two runs; 52 duplicates dropped (same problem, temperature, top-p and seed); 720 independent trajectories, 1,904,584 steps |

| Arm | Role | T | top-p | Problems | Hit the ceiling |
|---|---|---|---|---|---|
| cold-T0.1 | hypothesis | 0.1 | 1.00 | 124 | 29 (23.4 %) |
| cold-T0.2 | hypothesis | 0.2 | 1.00 | 124 | 23 (18.5 %) |
| cold-T0.3 | hypothesis | 0.3 | 1.00 | 72 | 8 (11.1 %) |
| demo-T0.3 | demonstration | 0.3 | 1.00 | 52 | 8 (15.4 %) |
| reference | reference | 0.6 | 0.95 | 100 | 8 (8.0 %) |
| hyp-T0.7 | hypothesis | 0.7 | 1.00 | 124 | 2 (1.6 %) |
| hyp-T1.0 | hypothesis | 1.0 | 1.00 | 124 | 6 (4.8 %) |

The two arms at T = 0.3 cover complementary problem sets: `demo-T0.3` and the first 52
problems of `cold-T0.3` are bit-identical, so the duplicates are dropped from `cold-T0.3`.

## Reproduce the numbers and figures

CPU only, about five minutes, Python 3.11.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-analysis.txt
# download early-entropy-budget-exhaustion-trajectories-v1.zip from https://doi.org/10.5281/zenodo.24606040 and unzip it here;
# it unpacks to data/runs/20260806T212541Z-768873/ and data/runs/20260820T220044Z-9e47ab/
unzip early-entropy-budget-exhaustion-trajectories-v1.zip
bash reproduce.sh
```

`reproduce.sh` writes the printed tables to `results/`, the figures to `figures/`, and
compares both tables with `expected/`. Where each number of the preprint comes from:

| Preprint | Source |
|---|---|
| Corpus and de-duplication (Sect. 3.1–3.2) | `analyze_corpus.py`, sections 0–1 |
| Temperature-invariant reading (Sect. 3.3) | `paper_numbers.py`, sections 7–8 |
| Early in the trajectory (Sect. 4.1) | `analyze_corpus.py`, section 4; `paper_numbers.py`, sections 9, 10, 14 and 15 |
| Two regimes of exhaustion (Sect. 4.2) | `analyze_corpus.py`, sections 2, 3 and 5; `paper_numbers.py`, section 14 |
| Close to failure (Sect. 4.3) | `analyze_corpus.py`, section 6; `paper_numbers.py`, sections 12–13 |
| Event definitions (Sect. 5.2) | `paper_numbers.py`, section 11 |
| Figures 1–3 | `make_figures.py` |

## Regenerate the trajectories

This needs a CUDA GPU with bf16 and 12 GB of memory; the corpus was generated on an
RTX 3060. Install the environment as described at the top of `environment.yml`
(`requirements.txt` holds the pinned generation stack, `env.lock.txt` the full freeze),
then:

```bash
python scripts/check_env.py         # calibrates the sensor on cases with exact answers
python scripts/run_smoke.py         # end-to-end run with fixed acceptance criteria
python scripts/check_batching.py    # batching must not change the measurement
python scripts/run_pilot.py         # run 1: reference, hyp-T0.7, hyp-T1.0, demo-T0.3
python scripts/run_cold_arms.py     # run 2: cold-T0.1, cold-T0.2, cold-T0.3
```

Runs are written to `data/runs/`. On the same machine and software versions,
regeneration is bit-identical: the 52 duplicate pairs above were generated two weeks
apart and agree in every token id, entropy and top-32 logit. On other hardware, expect token sequences to diverge at
near-ties in the logits, which changes individual trajectories but not the design.

## Data

Each run directory holds `config.yaml`, `env.lock.txt`, `git_commit`, and Parquet
shards:

- `trajectories-*.parquet`: one row per trajectory. Arm, temperature, top-p, seed,
  number of steps, termination (`eos` or `max_tokens`), extracted and gold answer,
  extraction outcome, correctness.
- `steps-*.parquet`: one row per decoding step. Sampled token id, the per-step signals
  listed under Setup, and the top-32 ids and logits.

Texts are not stored, only token ids; decode them with the model's tokenizer at the
pinned revision. The duplicates are kept in the release and dropped by the analysis.

`git_commit` refers to the development history this repository was cut from. The
generation code (`decoding.py`, `signals.py`, `runner.py`, `data.py`) has not changed
since either run; later commits added analysis helpers and column selection on read.
`run_cold_arms.py` was put under version control after run 2; its model, revision,
arms and batch size match that run's `config.yaml`.

## Labelling

Classes come from how a trajectory ended and whether its answer is correct, never from
the signal under test or from the text surface.

| | ended with EOS | hit the step ceiling |
|---|---|---|
| answer correct | `HEALTHY` | `DEGENERATE_LUCKY` (counted, excluded) |
| answer incorrect | `SEMANTIC_ERROR` | `FAILED_TERMINATION` |
| answer unscorable | `EXTRACTION_FAILED` (counted, excluded) | `FAILED_TERMINATION` |

Repetition and periodicity markers (`labeling.py`, and the period-agnostic marker in
`paper_numbers.py`) only locate the onset of a failure. Defining a class by repetition
would be circular, because repetition implies a peaked, low-entropy distribution.

Code comments refer to three hypotheses fixed before the runs:

- **H1a, observability.** The raw-logit signals separate `FAILED_TERMINATION` from
  `HEALTHY`.
- **H1b, lead time.** The signal changes before the failure onset, early enough to act on.
- **H1c, specificity.** The signal does not separate `SEMANTIC_ERROR` from `HEALTHY`;
  semantic errors are the negative control.

Arm roles decide what may be tested where: hypothesis arms carry H1a–c, the reference
arm sets baselines, and the demonstration arm reproduces a regime found earlier.

## Layout

```
src/cotctl/
  decoding.py   decode loop: raw logits -> signals -> sampling
  signals.py    per-step signals at the reference temperature
  runner.py     orchestration over arms, problems and seeds
  data.py       GSM8K loading, prompt, answer extraction
  labeling.py   classes; repetition and periodicity onset markers
  storage.py    Parquet schema and writers
  runs.py       loading runs, trajectory identity, de-duplication
scripts/
  check_env.py, run_smoke.py, check_batching.py, diagnose_batch_precision.py
  run_pilot.py, run_cold_arms.py           generation
  analyze_corpus.py, paper_numbers.py      every number in the preprint
  make_figures.py                          Figures 1-3
expected/       reference output of the two analysis scripts
reproduce.sh    numbers, figures and the comparison with expected/
```

## Licence and citation

Code: MIT, see `LICENSE`. The preprint and the trajectories are released under
CC BY 4.0 on Zenodo. GSM8K, whose problem ids and gold answers appear in the data, is
released under the MIT licence.

To cite, use `CITATION.cff` or the preprint's DOI, [10.5281/zenodo.24605925](https://doi.org/10.5281/zenodo.24605925).
