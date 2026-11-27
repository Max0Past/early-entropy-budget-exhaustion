#!/usr/bin/env python
"""Why does batching move the entropy? Padding leak, or bf16 reduction order?

`check_batching.py` fails at |dH| up to 7.3e-02 against a 1e-3 tolerance, and one
trajectory is token-identical over all 384 steps while still disagreeing by 0.073
nats. Identical tokens means identical context, so the sampler is not the cause:
the logits themselves differ. Two candidates, with opposite consequences.

  * **Padding leaks into the logits.** A bug. Batching is unusable until fixed,
    and this failure is dangerous precisely because the output stays
    plausible.
  * **Batched GEMM reduces in a different order.** Not a bug, and not fixable:
    bf16 carries ~8 mantissa bits, so a logit near 20 has a ULP around 0.06 --
    the same order as the observed discrepancy. Then the number to report is the
    sensor's precision under batching, and the 1e-3 tolerance was written without
    measuring anything.

The two are separable at **step 0**, before any sampling divergence can
accumulate:

  A. one prompt alone (batch 1)                     -- the reference
  B. the same prompt repeated 4x (batch 4)          -- no padding at all, since
                                                       every row is equal length
  C. four different prompts (batch 4)               -- real left padding
  D. C again with the row order reversed            -- same padding, different
                                                       position in the batch

B vs A isolates reduction order. C vs A adds padding on top. D vs C asks whether
the result depends on *where* a row sits, which nothing but a leak explains.

Usage:
    python scripts/diagnose_batch_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from cotctl.data import format_prompt, load_gsm8k  # noqa: E402
from cotctl.runner import load_model  # noqa: E402
from cotctl.signals import tempered_entropy  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
N_PROMPTS = 4


@torch.no_grad()
def prefill_logits(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    """Raw next-token logits after the prompt, one row per prompt, float32."""
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side="left"
    ).to(model.device)
    attention_mask = encoded["attention_mask"]
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    out = model(
        input_ids=encoded["input_ids"],
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=1,
    )
    return out.logits[:, -1, :].float()


def compare(name: str, reference: torch.Tensor, other: torch.Tensor) -> None:
    delta = (reference - other).abs()
    h_ref = tempered_entropy(reference, 1.0)
    h_other = tempered_entropy(other, 1.0)
    dh = (h_ref - h_other).abs()
    print(f"  {name:<34}"
          f"max|dlogit|={delta.max():.4e}  "
          f"max|dH|={dh.max():.4e}  mean|dH|={dh.mean():.4e}")


def main() -> int:
    model, tokenizer = load_model(MODEL, REVISION)
    problems = load_gsm8k("test", N_PROMPTS)
    prompts = [format_prompt(p.question, tokenizer) for p in problems]

    lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    print(f"prompt token counts: {lengths}")
    print(f"padding needed in a batch of {N_PROMPTS}: "
          f"{[max(lengths) - n for n in lengths]}\n")

    print("=== A. reference: each prompt alone, batch 1 ===")
    alone = torch.cat([prefill_logits(model, tokenizer, [p]) for p in prompts])
    print(f"  logit magnitude: max|z|={alone.abs().max():.2f}  "
          f"top-1 mean={alone.max(dim=-1).values.mean():.2f}")
    # bf16 keeps ~8 mantissa bits, so the spacing at a magnitude of |z| is
    # 2^(floor(log2|z|) - 7).
    top1 = float(alone.max(dim=-1).values.mean())
    ulp = 2.0 ** (int(torch.log2(torch.tensor(abs(top1))).floor()) - 7)
    print(f"  bf16 ULP at that magnitude: {ulp:.4f}")

    print("\n=== B. same prompt x4: no padding, batched GEMM only ===")
    for i, prompt in enumerate(prompts):
        repeated = prefill_logits(model, tokenizer, [prompt] * N_PROMPTS)
        rows_agree = (repeated - repeated[0]).abs().max()
        compare(f"prompt {i} vs batch-1", alone[i:i + 1], repeated[0:1])
        print(f"    (rows within the batch differ by {rows_agree:.2e} "
              "-- identical input, so this is 0 or it is a leak)")

    print("\n=== C. four different prompts: real left padding ===")
    mixed = prefill_logits(model, tokenizer, prompts)
    compare("batch-4 mixed vs batch-1", alone, mixed)

    print("\n=== D. same batch, rows reversed ===")
    reversed_batch = prefill_logits(model, tokenizer, prompts[::-1]).flip(0)
    compare("reversed vs forward order", mixed, reversed_batch)
    compare("reversed vs batch-1", alone, reversed_batch)
    print("\n  If D moves with position, padding is leaking. If C and D are the")
    print("  same size as B, the whole effect is reduction order in bf16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
