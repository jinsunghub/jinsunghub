# Speculative Decoding Experiment Report

## 1. Objective
- Measure how `k` (draft proposal length per round) affects acceptance and efficiency in:
  - `speculative-decoding-lab/spec_decode_toy.py`
  - `speculative-decoding-lab/spec_decode_hf.py`
- Explain the observed behavior from the actual implementation logic.

## 2. Setup
- Date: 2026-02-13
- Environment: Windows + Python, CPU inference
- Prompt:
  - Toy: `"I"`
  - HF: `"The future of AI is"`

### Toy runs
- `max-new-tokens=30`
- `k=1..8`
- Seeds: `0..49` (50 runs per `k`), report mean values

### HF runs
- `max-new-tokens=20`
- Target model: `gpt2`
- Draft model: `sshleifer/tiny-gpt2`
- `k in {1,2,4,8}`

### Additional validation
- HF with identical models:
  - Target = Draft = `sshleifer/tiny-gpt2`

## 3. Results

### 3.1 Toy summary (50-seed mean)

| k | mean acceptance | mean target calls | mean proposed | mean accepted |
|---|---:|---:|---:|---:|
| 1 | 0.4407 | 30.00 | 30.00 | 13.22 |
| 2 | 0.3128 | 30.12 | 41.64 | 12.86 |
| 3 | 0.2485 | 30.24 | 54.78 | 13.28 |
| 4 | 0.1892 | 30.38 | 71.68 | 13.00 |
| 5 | 0.1592 | 30.32 | 87.30 | 13.12 |
| 6 | 0.1266 | 30.28 | 105.84 | 12.74 |
| 7 | 0.1258 | 30.42 | 115.36 | 13.94 |
| 8 | 0.1056 | 30.50 | 134.56 | 13.70 |

Key trend:
- Increasing `k` lowers acceptance.
- `accepted` stays around ~13 while `proposed` increases sharply.

### 3.2 HF summary (`gpt2` vs `tiny-gpt2`)

| k | elapsed | target calls | proposed | accepted | acceptance |
|---|---:|---:|---:|---:|---:|
| 1 | 1.38s | 20 | 20 | 0 | 0.00% |
| 2 | 1.37s | 20 | 39 | 0 | 0.00% |
| 4 | 1.46s | 20 | 74 | 0 | 0.00% |
| 8 | 1.57s | 20 | 132 | 0 | 0.00% |

Key trend:
- Acceptance remains 0% for all `k`.
- Larger `k` only increases discarded proposals.

### 3.3 HF identical-model sanity check
- Target = Draft = `sshleifer/tiny-gpt2`
- Example (`max-new-tokens=10`, `k=4`): acceptance = `100%`

This confirms implementation behavior is consistent with the acceptance rule.

## 4. Why this happens

### 4.1 Acceptance rule is strict token equality
In `spec_decode_hf.py`, a draft token is accepted only when it exactly matches the target greedy token at that step.

### 4.2 `gpt2` and `tiny-gpt2` next-token behavior is very different
Observed first 5 greedy next tokens for prompt `"The future of AI is"`:
- `gpt2`: `uncertain`, `.`, `The`, `future`, `of`
- `sshleifer/tiny-gpt2`: `stairs`, `stairs`, `stairs`, `stairs`, `stairs`

Because the very first prediction usually mismatches, each round rejects immediately.

### 4.3 Early break on mismatch limits effective speculative gain
Both toy and HF scripts stop proposal verification at first mismatch and append target token correction.
So with poor draft-target alignment:
- `accepted` does not increase much,
- `proposed` keeps growing with larger `k`,
- acceptance (`accepted/proposed`) drops.

### 4.4 Why target calls stay near max-new-tokens
Mismatch still appends one target token, so generation continues one token at a time.
This keeps `target_calls` near `max-new-tokens` regardless of larger `k`.

## 5. Implementation update included in this work
- File: `speculative-decoding-lab/spec_decode_hf.py`
- Fix: enforce exact `max_new_tokens` output length.
- Change: replaced loop logic with `new_tokens`-tracked control so generation does not exceed the requested token budget.

## 6. Conclusion
Speculative decoding speedup depends on draft-target agreement.
- If draft approximates target well: high acceptance, potential speedup.
- If draft is poorly aligned (as with `gpt2` vs `tiny-gpt2` here): acceptance collapses and larger `k` mostly increases wasted proposals.

## 7. Repro Commands

```bash
# Toy single run
python speculative-decoding-lab/spec_decode_toy.py --prompt "I" --max-new-tokens 30 --k 4 --seed 7

# HF single run
python speculative-decoding-lab/spec_decode_hf.py --target-model gpt2 --draft-model sshleifer/tiny-gpt2 --prompt "The future of AI is" --max-new-tokens 20 --k 4

# HF sanity check (identical models)
python speculative-decoding-lab/spec_decode_hf.py --target-model sshleifer/tiny-gpt2 --draft-model sshleifer/tiny-gpt2 --prompt "The future of AI is" --max-new-tokens 10 --k 4
```
