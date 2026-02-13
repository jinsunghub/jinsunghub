"""Toy speculative decoding demo without external model dependencies."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

VOCAB = ["I", "you", "we", "like", "love", "build", "AI", "systems", ".", "\n"]
IDX = {tok: i for i, tok in enumerate(VOCAB)}


@dataclass
class Stats:
    target_calls: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0


def sample_from_probs(rng: random.Random, probs: list[float]) -> int:
    x = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if x <= cum:
            return i
    return len(probs) - 1


def argmax_index(probs: list[float]) -> int:
    return max(range(len(probs)), key=lambda i: probs[i])


class ToyLM:
    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    def probs(self, prev_token: str) -> list[float]:
        return self.table.get(prev_token, self.table["<default>"])


# target는 문법적으로 더 일관된 분포를 가정
TARGET_TABLE = {
    "<default>": [0.12, 0.10, 0.08, 0.16, 0.08, 0.14, 0.12, 0.08, 0.10, 0.02],
    "I": [0.01, 0.04, 0.03, 0.38, 0.30, 0.17, 0.03, 0.01, 0.02, 0.01],
    "you": [0.02, 0.01, 0.02, 0.35, 0.25, 0.26, 0.04, 0.01, 0.03, 0.01],
    "we": [0.02, 0.02, 0.01, 0.22, 0.20, 0.35, 0.08, 0.02, 0.06, 0.02],
    "like": [0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.55, 0.25, 0.06, 0.01],
    "love": [0.01, 0.01, 0.01, 0.02, 0.02, 0.05, 0.50, 0.28, 0.09, 0.01],
    "build": [0.01, 0.01, 0.01, 0.01, 0.01, 0.06, 0.30, 0.45, 0.12, 0.02],
    "AI": [0.03, 0.02, 0.02, 0.07, 0.05, 0.30, 0.08, 0.34, 0.08, 0.01],
    "systems": [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.04, 0.83, 0.01],
    ".": [0.28, 0.21, 0.20, 0.03, 0.03, 0.04, 0.05, 0.03, 0.01, 0.12],
}

# draft는 target을 대충 근사하지만 약간 noisy한 분포
DRAFT_TABLE = {
    "<default>": [0.11, 0.11, 0.09, 0.15, 0.09, 0.12, 0.12, 0.10, 0.09, 0.02],
    "I": [0.02, 0.05, 0.04, 0.34, 0.26, 0.20, 0.04, 0.01, 0.03, 0.01],
    "you": [0.03, 0.02, 0.03, 0.30, 0.23, 0.27, 0.05, 0.01, 0.05, 0.01],
    "we": [0.03, 0.03, 0.02, 0.20, 0.19, 0.33, 0.10, 0.03, 0.06, 0.01],
    "like": [0.01, 0.01, 0.01, 0.03, 0.03, 0.08, 0.47, 0.28, 0.07, 0.01],
    "love": [0.01, 0.01, 0.01, 0.02, 0.03, 0.06, 0.45, 0.30, 0.10, 0.01],
    "build": [0.01, 0.01, 0.01, 0.01, 0.01, 0.09, 0.27, 0.42, 0.14, 0.03],
    "AI": [0.03, 0.03, 0.02, 0.07, 0.06, 0.28, 0.10, 0.31, 0.09, 0.01],
    "systems": [0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.05, 0.06, 0.80, 0.01],
    ".": [0.30, 0.22, 0.21, 0.03, 0.03, 0.04, 0.03, 0.03, 0.01, 0.10],
}


def speculative_decode(
    prompt: str,
    max_new_tokens: int,
    k: int,
    seed: int,
) -> tuple[list[str], Stats]:
    rng = random.Random(seed)
    target = ToyLM(TARGET_TABLE)
    draft = ToyLM(DRAFT_TABLE)

    tokens = prompt.split()
    if not tokens:
        tokens = ["I"]

    stats = Stats()

    while len(tokens) < max_new_tokens + 1:
        # 1) draft가 k개 제안
        proposal: list[int] = []
        ctx = tokens[-1]
        for _ in range(k):
            d_probs = draft.probs(ctx)
            d_idx = sample_from_probs(rng, d_probs)
            proposal.append(d_idx)
            ctx = VOCAB[d_idx]

        stats.proposed_tokens += len(proposal)

        # 2) target이 순차 검증
        accepted = 0
        ctx = tokens[-1]
        for d_idx in proposal:
            t_probs = target.probs(ctx)
            stats.target_calls += 1
            t_idx = argmax_index(t_probs)  # 단순화를 위해 greedy 검증
            if d_idx == t_idx:
                tokens.append(VOCAB[d_idx])
                accepted += 1
                ctx = VOCAB[d_idx]
            else:
                tokens.append(VOCAB[t_idx])
                ctx = VOCAB[t_idx]
                break

        stats.accepted_tokens += accepted

        if len(tokens) >= max_new_tokens + 1:
            break

    return tokens, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="I")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokens, stats = speculative_decode(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        k=args.k,
        seed=args.seed,
    )

    text = " ".join(tokens)
    acceptance = (
        stats.accepted_tokens / stats.proposed_tokens if stats.proposed_tokens else 0.0
    )

    print("=== Speculative Decoding (Toy) ===")
    print(f"prompt: {args.prompt}")
    print(f"k: {args.k}")
    print(f"generated: {text}")
    print(f"target calls: {stats.target_calls}")
    print(f"proposed tokens: {stats.proposed_tokens}")
    print(f"accepted tokens: {stats.accepted_tokens}")
    print(f"acceptance rate: {acceptance:.2%}")


if __name__ == "__main__":
    main()
