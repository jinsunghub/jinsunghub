"""Minimal Hugging Face speculative decoding playground.

Note: This is a simplified educational implementation.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def next_token_greedy(model, input_ids):
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    return torch.argmax(logits, dim=-1, keepdim=True)


def speculative_decode_hf(
    target_model,
    draft_model,
    input_ids,
    max_new_tokens: int,
    k: int,
):
    generated = input_ids.clone()
    proposed = 0
    accepted = 0
    target_calls = 0
    new_tokens = 0

    while new_tokens < max_new_tokens:
        # Draft proposes up to k tokens, but never beyond max_new_tokens.
        proposal = []
        draft_ctx = generated.clone()
        remaining = max_new_tokens - new_tokens
        for _ in range(min(k, remaining)):
            d_next = next_token_greedy(draft_model, draft_ctx)
            proposal.append(d_next)
            draft_ctx = torch.cat([draft_ctx, d_next], dim=-1)
            proposed += 1

        # target verifies in order
        for d_next in proposal:
            if new_tokens >= max_new_tokens:
                break
            t_next = next_token_greedy(target_model, generated)
            target_calls += 1
            if torch.equal(d_next, t_next):
                generated = torch.cat([generated, d_next], dim=-1)
                accepted += 1
            else:
                generated = torch.cat([generated, t_next], dim=-1)
                new_tokens += 1
                break
            new_tokens += 1

    return generated, {"proposed": proposed, "accepted": accepted, "target_calls": target_calls}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="gpt2")
    parser.add_argument("--draft-model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(args.target_model)
    draft = AutoModelForCausalLM.from_pretrained(args.draft_model)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    start = time.time()
    out_ids, metrics = speculative_decode_hf(
        target_model=target,
        draft_model=draft,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        k=args.k,
    )
    elapsed = time.time() - start

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    acceptance = metrics["accepted"] / max(1, metrics["proposed"])

    print("=== Speculative Decoding (HF) ===")
    print(f"target model: {args.target_model}")
    print(f"draft model: {args.draft_model}")
    print(f"k: {args.k}")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"target calls: {metrics['target_calls']}")
    print(f"proposed: {metrics['proposed']}")
    print(f"accepted: {metrics['accepted']}")
    print(f"acceptance rate: {acceptance:.2%}")
    print("\n--- output ---")
    print(text)


if __name__ == "__main__":
    main()
