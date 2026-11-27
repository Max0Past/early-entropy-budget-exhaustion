"""Custom decoding loop with per-step state logging.

This loop, not `model.generate` + a LogitsProcessor, is the harness for Phase I.
Reason: processor ordering is not contractually guaranteed across transformers
versions, so there is no version-stable way to assert that what you observe are
the raw pre-temperature logits. Here the sensor reads `outputs.logits` directly,
before anything touches them.

The same loop becomes the Phase-II control loop unchanged: the only edit is that
`temperature` stops being a constant and starts coming from the regulator.

Phase-I invariants, all of them load-bearing:
    * No repetition_penalty. No no_repeat_ngram. Those mask the exact regime
      under study; a run with them enabled is worthless.
    * top_p = 1.0 on the failure-induction arms, so that the plant is pure
      temperature sampling and nothing else.
    * Signals are computed at T_ref = 1 regardless of the sampling temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cotctl.signals import (
    N_SIGNALS,
    TEMPERATURE_GRID,
    compute_step_signals,
    compute_step_signals_batched,
    count_reachable_band_violations,
)


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    """One arm of the sweep."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    # Temperatures at which the sensor evaluates H each step. Defaults to the
    # grid in signals.py, which spans and overshoots the admissible band.
    grid: tuple[float, ...] = TEMPERATURE_GRID
    top_k_log: int = 32     # how many (id, logit) pairs to persist per step, as insurance


@dataclass(slots=True)
class Trajectory:
    """One generated trajectory plus its full observation record."""

    token_ids: list[int]
    signals: torch.Tensor          # (n_steps, N_SIGNALS), float32, on CPU
    topk_ids: torch.Tensor         # (n_steps, top_k_log), int32, on CPU
    topk_logits: torch.Tensor      # (n_steps, top_k_log), float32, on CPU
    termination: str               # "eos" | "max_tokens"
    text: str = ""
    meta: dict = field(default_factory=dict)


def _sample_batched(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Draw one token per row. The actuator, batched. `logits` is (B, V).

    **One generator per row, not one per batch.** With a shared generator the
    draws for row `i` depend on how many rows sit beside it, so a trajectory's
    tokens would depend on which other problems happened to be batched with it.
    That is not a testing inconvenience, it is a reproducibility defect: the same
    (problem, seed) would give different text at different batch sizes, and no
    run could be reproduced without also reproducing its batching.

    Per-row generators make batch size invisible to the sampler, which is what
    lets the regression test attribute any divergence to the forward pass.

    The cost is `B` small `multinomial` calls per step instead of one batched
    call. Against a 72 ms forward pass on this hardware that is noise, and the
    loop is CPU-launch-bound anyway.
    """
    scaled = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        sorted_logits = sorted_logits.masked_fill(mass_before > top_p, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter_(
            1, sorted_idx, sorted_logits
        )
    probs = torch.softmax(scaled, dim=-1)
    drawn = [
        torch.multinomial(probs[i], num_samples=1, generator=generators[i])
        for i in range(probs.shape[0])
    ]
    return torch.cat(drawn)


@torch.no_grad()
def generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    cfg: DecodeConfig,
    generators: list[torch.Generator],
) -> list[Trajectory]:
    """Generate several trajectories at once, returning them in input order.

    A second function rather than a generalisation of `generate_trajectory`. The
    single-stream loop stays byte-for-byte the reference the regression test
    compares against; a test whose reference shares an implementation with the
    thing under test proves nothing.

    **Left padding, always.** With right padding the last real token sits at a
    different index in every row and there is no single position to read logits
    from. Left padding puts it at -1 for every row. `position_ids` must then be
    derived from the attention mask, or padded rows are given positions that
    never existed and the padding leaks into the logits -- the failure
    the pipeline notes rank fifth and describe as producing entirely plausible
    output.

    Rows that hit EOS stop contributing: their signals are no longer recorded and
    they are fed a pad token to keep the batch rectangular. Generation ends when
    every row has finished or the ceiling is reached.
    """
    eos_ids = _eos_token_ids(model, tokenizer)
    device = model.device
    batch = len(prompts)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side="left"
    ).to(device)
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]

    # Padded rows must not be told they start at position 0.
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        logits_to_keep=1,
    )
    past = outputs.past_key_values
    logits = outputs.logits[:, -1, :].float()
    next_position = position_ids[:, -1] + 1

    signals_buf: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    topk_ids_buf: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    topk_logits_buf: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    token_ids: list[list[int]] = [[] for _ in range(batch)]
    terminations = ["max_tokens"] * batch
    prev_log_p: torch.Tensor | None = None
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    eos_tensor = torch.tensor(sorted(eos_ids), device=device)

    for _ in range(cfg.max_new_tokens):
        tokens = _sample_batched(
            logits, cfg.temperature, cfg.top_p, generators)
        step_signals, prev_log_p = compute_step_signals_batched(
            logits, tokens, prev_log_p, cfg.grid
        )
        top = torch.topk(logits, k=cfg.top_k_log, dim=-1)

        hit_eos = torch.isin(tokens, eos_tensor)
        active = (~finished).tolist()  # one sync per step, as in the reference
        token_list = tokens.tolist()

        for i in range(batch):
            if not active[i]:
                continue
            signals_buf[i].append(step_signals[i])
            topk_ids_buf[i].append(top.indices[i].to(torch.int32))
            topk_logits_buf[i].append(top.values[i])
            token_ids[i].append(token_list[i])

        finished |= hit_eos
        for i in range(batch):
            if active[i] and hit_eos[i]:
                terminations[i] = "eos"
        if bool(finished.all()):
            break

        # Finished rows are fed padding so the batch stays rectangular; their
        # outputs are discarded above.
        tokens = torch.where(finished, torch.full_like(tokens, pad_id), tokens)
        attention_mask = torch.cat(
            [attention_mask, (~finished).long().unsqueeze(1)], dim=-1
        )
        outputs = model(
            input_ids=tokens.unsqueeze(1),
            attention_mask=attention_mask,
            position_ids=next_position.unsqueeze(1),
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()
        next_position = next_position + 1

    trajectories: list[Trajectory] = []
    for i in range(batch):
        signals = (
            torch.stack(signals_buf[i]).cpu()
            if signals_buf[i]
            else torch.empty(0, N_SIGNALS)
        )
        violations = count_reachable_band_violations(signals, grid=cfg.grid)
        if violations:
            raise RuntimeError(
                f"H(T) is not monotone at {violations}/{len(signals)} steps in "
                f"batched row {i}. Padding leaking into the logits looks exactly "
                "like this."
            )
        trajectories.append(
            Trajectory(
                token_ids=token_ids[i],
                signals=signals,
                topk_ids=torch.stack(topk_ids_buf[i]).cpu(),
                topk_logits=torch.stack(topk_logits_buf[i]).cpu(),
                termination=terminations[i],
                text=tokenizer.decode(token_ids[i], skip_special_tokens=True),
                meta={
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_new_tokens": cfg.max_new_tokens,
                    "n_steps": len(token_ids[i]),
                    "batch_size": batch,
                },
            )
        )
    return trajectories


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter. Keeps the smallest prefix whose cumulative mass reaches top_p.

    Always keeps at least the top-1 token: the mask compares the cumulative mass
    STRICTLY BEFORE each token, so the first token can never be removed.
    """
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs

    sorted_logits = sorted_logits.masked_fill(mass_before > top_p, float("-inf"))

    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, sorted_idx, sorted_logits)
    return filtered


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one token. This is the ACTUATOR: the only place temperature acts."""
    scaled = logits / temperature
    if top_p < 1.0:
        scaled = _apply_top_p(scaled, top_p)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)[0]


def _eos_token_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Qwen-family models terminate on more than one token; handle the list form."""
    raw = model.generation_config.eos_token_id
    if raw is None:
        raw = tokenizer.eos_token_id
    if isinstance(raw, int):
        return {raw}
    return set(raw)


@torch.no_grad()
def generate_trajectory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    cfg: DecodeConfig,
    generator: torch.Generator,
) -> Trajectory:
    """Generate one trajectory, logging the state signals at every step."""
    eos_ids = _eos_token_ids(model, tokenizer)
    device = model.device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Prefill would otherwise return logits for EVERY prompt position, i.e.
    # batch x prompt_len x 152k -- about 1.9 GB at batch 16 with a 400-token
    # prompt, allocated before a single token has been generated. We need only
    # the last position.
    #
    # The kept row is NOT bitwise equal to the one we would have sliced out:
    # measured ~2.4e-07, i.e. a couple of float32 ULP, because narrowing the
    # lm_head GEMM changes its reduction order. Same phenomenon the pipeline notes
    # predicts for batch=1 vs batch=N. It sits ~5 orders of magnitude below the
    # bf16 sensor floor (7.6e-03 nats), so it is irrelevant to the measurement,
    # but runs made with and without this kwarg are not bitwise comparable.
    #
    # The kwarg was renamed from `num_logits_to_keep`; scripts/check_env.py
    # asserts which spelling the pinned transformers actually exposes.
    outputs = model(**inputs, use_cache=True, logits_to_keep=1)
    past = outputs.past_key_values
    logits = outputs.logits[0, -1, :].float()  # bf16 -> fp32; mandatory, see signals.py

    signals_buf: list[torch.Tensor] = []
    topk_ids_buf: list[torch.Tensor] = []
    topk_logits_buf: list[torch.Tensor] = []
    token_ids: list[int] = []
    prev_log_p: torch.Tensor | None = None
    termination = "max_tokens"

    for _ in range(cfg.max_new_tokens):
        token = sample_token(logits, cfg.temperature, cfg.top_p, generator)

        step_signals, prev_log_p = compute_step_signals(
            logits, token, prev_log_p, cfg.grid
        )
        signals_buf.append(step_signals)

        top = torch.topk(logits, k=cfg.top_k_log)
        topk_ids_buf.append(top.indices.to(torch.int32))
        topk_logits_buf.append(top.values)

        token_id = int(token)  # the one unavoidable host sync per step: the EOS check
        token_ids.append(token_id)
        if token_id in eos_ids:
            termination = "eos"
            break

        outputs = model(
            input_ids=token.view(1, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        logits = outputs.logits[0, -1, :].float()

    # One device->host transfer per trajectory, not per step.
    signals = torch.stack(signals_buf).cpu() if signals_buf else torch.empty(0, N_SIGNALS)

    # dH/dT > 0, so H must increase monotonically across the temperature grid at
    # every step. Free to check here, and across seven points it constrains the
    # whole curve rather than just its endpoints.
    violations = count_reachable_band_violations(signals, grid=cfg.grid)
    if violations:
        raise RuntimeError(
            f"H(T) is not monotone at {violations}/{len(signals)} steps over "
            f"grid {cfg.grid}. dH/dT > 0, so this means a swapped bound, an "
            "inverted sign, or a grid that is not sorted ascending."
        )

    return Trajectory(
        token_ids=token_ids,
        signals=signals,
        topk_ids=torch.stack(topk_ids_buf).cpu(),
        topk_logits=torch.stack(topk_logits_buf).cpu(),
        termination=termination,
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        meta={
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_new_tokens": cfg.max_new_tokens,
            "n_steps": len(token_ids),
            "prompt_len": int(inputs["input_ids"].shape[1]),
        },
    )
