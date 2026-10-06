#!/usr/bin/env python3
"""
Run the ToolDefer conditions on a local MLX model.

Each prompt is scored with a single forward pass: the next-token logits for the
two answer tokens ("1"/"2" or "Yes"/"No") give the verdict and a continuous
approval probability. Nothing is generated, so results are deterministic.

Usage:
    python tooldefer_run_local.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
    python tooldefer_run_local.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --max-rows 5
    python tooldefer_run_local.py --model ... --generate-check 100
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import append_jsonl, extract_answer_from_text, load_jsonl
from tooldefer_prompts import ANSWER_FORMATS, ANSWER_ORDERS, CONDITIONS, answer_tokens, build_prompt, expand_record


def encode(tokenizer, prompt: str) -> list:
    """Apply the chat template to a single user message."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )


def answer_token_id(tokenizer, text: str) -> int:
    """Return the token id for an answer string, which must be a single token."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"Answer {text!r} is not a single token for this tokenizer: {encoded}")
    return encoded[0]


def score_prompt(model, tokenizer, prompt: str, approve_text: str, reject_text: str) -> dict:
    """Score one prompt and return the approval probability over the two answer tokens."""
    tokens = encode(tokenizer, prompt)
    logits = model(mx.array([tokens]))[0, -1, :]
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    p_yes = probs[answer_token_id(tokenizer, approve_text)].item()
    p_no = probs[answer_token_id(tokenizer, reject_text)].item()
    mass = p_yes + p_no
    return {
        "p_approve": p_yes / mass if mass > 0 else 0.5,
        "approve": int(p_yes > p_no),
        "answer_mass": mass, "n_tokens": len(tokens),
    }


def run_experiment(args) -> None:
    records = load_jsonl(args.input)
    if args.max_rows is not None:
        records = records[:args.max_rows]

    done = set()
    if Path(args.output).exists():
        done = {(r["id"], r["variant"], r["condition"], r["order"]) for r in load_jsonl(args.output)}

    # Prompts vary in length, so MLX's buffer cache otherwise grows until the machine swaps.
    mx.set_cache_limit(1024 ** 3)

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    total = sum(len(expand_record(r)) for r in records) * len(CONDITIONS) * len(ANSWER_ORDERS)
    print(f"Items: {len(records)} | prompts: {total} | already done: {len(done)}")

    n_scored, start = 0, time.time()
    for record in records:
        for variant, truth, call in expand_record(record):
            for condition in CONDITIONS:
                for order in ANSWER_ORDERS:
                    if (record["id"], variant, condition, order) in done:
                        continue
                    prompt = build_prompt(condition, record["question"], record["functions"], call,
                                          order, args.answer_format)
                    scores = score_prompt(model, tokenizer, prompt, *answer_tokens(order, args.answer_format))
                    append_jsonl({
                        "id": record["id"], "dataset": record["dataset"], "model": args.model,
                        "answer_format": args.answer_format,
                        "variant": variant, "truth": truth, "condition": condition, "order": order, **scores,
                    }, args.output)
                    n_scored += 1
                    if n_scored % 500 == 0:
                        rate = n_scored / (time.time() - start)
                        print(f"  {n_scored} scored | {rate:.1f} prompts/s | ~{(total - len(done) - n_scored) / rate / 60:.0f} min left")
    print(f"Scoring complete. Results saved to {args.output}")

    if args.generate_check:
        run_generation_check(model, tokenizer, records[:args.generate_check], args)


def run_generation_check(model, tokenizer, records, args) -> None:
    """Compare the logit verdict with a generated answer on a subset (C1 and C2, yes_first)."""
    output = args.output.replace(".jsonl", "_gencheck.jsonl")
    agree = n = 0
    for record in records:
        for variant, truth, call in expand_record(record)[:2]:
            for condition in ("c1", "c2"):
                prompt = build_prompt(condition, record["question"], record["functions"], call,
                                      "yes_first", args.answer_format)
                approve_text, reject_text = answer_tokens("yes_first", args.answer_format)
                scores = score_prompt(model, tokenizer, prompt, approve_text, reject_text)
                logit_answer = approve_text if scores["approve"] else reject_text
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
                )
                reply = generate(model, tokenizer, prompt=formatted, max_tokens=8)
                if args.answer_format == "yesno":
                    first_word = reply.strip().split()[0].strip(".,!").capitalize() if reply.strip() else None
                    generated_answer = first_word if first_word in ("Yes", "No") else None
                else:
                    generated_answer, _ = extract_answer_from_text(reply)
                n += 1
                agree += int(generated_answer == logit_answer)
                append_jsonl({
                    "id": record["id"], "variant": variant, "condition": condition,
                    "logit_answer": logit_answer, "generated_answer": generated_answer, "reply": reply,
                }, output)
    print(f"Generation check: {agree}/{n} generated answers match the logit verdict ({100 * agree / n:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run ToolDefer conditions on a local MLX model")
    parser.add_argument("--input", "-i", default="dataset/tooldefer/bfcl_pairs.jsonl", help="Pair dataset JSONL")
    parser.add_argument("--model", "-m", required=True, help="MLX model repo or local path")
    parser.add_argument("--output", "-o", help="Output JSONL path (auto-generated if not specified)")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="numbered",
                        help="numbered = answer 1 or 2 (DialDefer style); yesno = answer Yes or No")
    parser.add_argument("--max-rows", type=int, help="Maximum items to process (for testing)")
    parser.add_argument("--generate-check", type=int, default=0,
                        help="Also generate answers for the first N items and compare with the logit verdict")
    args = parser.parse_args()

    if not args.output:
        args.output = f"results/tooldefer/bfcl_{args.model.split('/')[-1]}_{args.answer_format}_results.jsonl"
    run_experiment(args)


if __name__ == "__main__":
    main()
