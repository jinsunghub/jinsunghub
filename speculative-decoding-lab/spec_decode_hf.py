"""Colab-friendly speculative decoding demo for Llama-family models.

This script supports two modes:
1) Built-in speculative decoding via `assistant_model=...` in `generate`.
2) Baseline decoding without assistant model for speed comparison.

Designed for GPU usage (e.g., Google Colab T4/L4/A100).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name: str, use_4bit: bool, device_map: str = "auto"):
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    model.eval()
    return model


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7 if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.time() - start
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, elapsed


def generate_speculative(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    num_assistant_tokens: int,
    do_sample: bool,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(target_model.device)
    start = time.time()
    with torch.no_grad():
        outputs = target_model.generate(
            **inputs,
            assistant_model=draft_model,
            num_assistant_tokens=num_assistant_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7 if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.time() - start
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--draft-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--prompt", type=str, default="Write 5 practical tips for learning CUDA.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    # Backward compatibility: some older docs/scripts used --k.
    parser.add_argument("--num-assistant-tokens", type=int, default=None)
    parser.add_argument("--k", type=int, default=None, help="Alias of --num-assistant-tokens")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--sample", action="store_true", help="Enable sampling")
    args = parser.parse_args()

    num_assistant_tokens = args.num_assistant_tokens
    if num_assistant_tokens is None:
        num_assistant_tokens = args.k if args.k is not None else 8

    try:
        print("[1/4] Loading tokenizer from target model...")
        tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        use_4bit = not args.no_4bit

        print(f"[2/4] Loading target model: {args.target_model}")
        target = load_model(args.target_model, use_4bit=use_4bit)

        print(f"[3/4] Loading draft model:  {args.draft_model}")
        draft = load_model(args.draft_model, use_4bit=use_4bit)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "403" in msg:
            print("\n[ERROR] 접근 제한(gated) 모델이라 로드할 수 없습니다.")
            print("- 현재 계정에 접근 권한이 있는 모델을 쓰거나,")
            print("- 공개 모델 조합(기본값)으로 실행하세요.")
            print("예시:")
            print(
                "python spec_decode_hf.py --target-model Qwen/Qwen2.5-7B-Instruct "
                "--draft-model Qwen/Qwen2.5-0.5B-Instruct --k 8"
            )
            print("\nLlama 2를 꼭 쓰려면 HF에서 meta-llama/Llama-2-7b-hf 접근 승인 후 login()이 필요합니다.")
            sys.exit(1)
        raise

    print("[4/4] Running baseline vs speculative decoding...")
    baseline_text, baseline_time = generate_baseline(
        target,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        do_sample=args.sample,
    )

    spec_text, spec_time = generate_speculative(
        target,
        draft,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        num_assistant_tokens,
        do_sample=args.sample,
    )

    speedup = baseline_time / spec_time if spec_time > 0 else 0.0

    print("\n=== Result ===")
    print(f"assistant tokens:  {num_assistant_tokens}")
    print(f"baseline time:    {baseline_time:.2f}s")
    print(f"speculative time: {spec_time:.2f}s")
    print(f"speedup:          {speedup:.2f}x")

    print("\n--- Baseline Output ---")
    print(baseline_text)
    print("\n--- Speculative Output ---")
    print(spec_text)


if __name__ == "__main__":
    main()
