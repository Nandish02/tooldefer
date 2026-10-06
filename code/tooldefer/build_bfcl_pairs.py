#!/usr/bin/env python3
"""
Build matched correct/incorrect tool-call pairs from BFCL for the ToolDefer pilot.

Each BFCL item (user request + function schemas + ground-truth call) yields one
correct call and up to three incorrect calls, one per perturbation type:

    P1_wrong_value   a required argument takes a value outside the acceptable list (semantic)
    P3_drop_required a required argument is removed (schema-checkable)
    P4_extra_param   an argument that is not in the schema is added (schema-checkable)

Every call is checked against the BFCL ground truth: the correct call must pass
and every perturbed call must fail, otherwise the perturbation is discarded.

Usage:
    python build_bfcl_pairs.py
    python build_bfcl_pairs.py --raw-dir dataset/tooldefer/raw --output dataset/tooldefer/bfcl_pairs.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_jsonl, save_jsonl

BFCL_CATEGORIES = ["simple_python", "multiple"]
BFCL_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"

# Arguments that no pilot schema defines; used for P4.
EXTRA_PARAMS = [
    ("verbose", True),
    ("timeout", 30),
    ("output_format", "json"),
    ("strict_mode", True),
    ("max_retries", 3),
    ("language", "en"),
]


# ---
# GROUND-TRUTH CHECKING
# ---

def standardize_string(value: str) -> str:
    """Normalise a string the way BFCL does before comparing argument values."""
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", value).lower().replace("'", '"')


def _normalize(value: Any) -> Any:
    """Normalise a value recursively so equal-meaning values compare equal."""
    if isinstance(value, str):
        return standardize_string(value)
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _unwrap(value: Any) -> Any:
    """Turn a ground-truth dict ({key: [acceptable, ...]}) into a concrete dict."""
    if isinstance(value, dict):
        return {k: _unwrap(v[0]) if isinstance(v, list) and v else v for k, v in value.items()}
    return value


def value_is_acceptable(value: Any, acceptable: List[Any]) -> bool:
    """Check a concrete argument value against a ground-truth acceptable list."""
    target = _normalize(value)
    for option in acceptable:
        if isinstance(option, str) and option == "":
            continue
        if isinstance(option, bool) != isinstance(value, bool):
            continue
        if _normalize(_unwrap(option)) == target:
            return True
    return False


def is_valid_call(fname: str, args: Dict[str, Any], gt: Dict[str, Dict[str, list]], schema: dict) -> bool:
    """Return True if the call satisfies the BFCL ground truth for this item."""
    if fname not in gt:
        return False
    expected = gt[fname]
    properties = schema["parameters"]["properties"]
    required = schema["parameters"].get("required", [])

    for param in required:
        if param not in args:
            return False
    for param, value in args.items():
        if param not in properties or param not in expected:
            return False
        if not value_is_acceptable(value, expected[param]):
            return False
    for param, acceptable in expected.items():
        if param not in args and "" not in acceptable:
            return False
    return True


# ---
# CALL CONSTRUCTION
# ---

def build_correct_args(gt_args: Dict[str, list], schema: dict) -> Dict[str, Any]:
    """Build the canonical correct call: required args plus optional args that cannot be omitted."""
    properties = schema["parameters"]["properties"]
    args = {}
    for param in properties:
        if param not in gt_args:
            continue
        acceptable = gt_args[param]
        if "" in acceptable:
            continue
        args[param] = _unwrap(acceptable[0])
    return args


def render_call(fname: str, args: Dict[str, Any]) -> str:
    """Render a call as fname(arg=value, ...)."""
    return f"{fname}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


# ---
# PERTURBATIONS
# ---

def perturb_wrong_value(
    args: Dict[str, Any], gt_args: Dict[str, list], schema: dict,
    donors: Dict[str, List[str]], rng: random.Random
) -> Optional[Tuple[Dict[str, Any], str]]:
    """P1: replace one required scalar argument with a plausible value outside the acceptable list."""
    properties = schema["parameters"]["properties"]
    required = [p for p in schema["parameters"].get("required", []) if p in args]
    rng.shuffle(required)

    for param in required:
        value = args[param]
        acceptable = gt_args[param]
        candidates: List[Any] = []
        if isinstance(value, bool):
            candidates = [not value]
        elif isinstance(value, int):
            steps = [1, 2, 5] if abs(value) < 10 else [1, 5, 10]
            candidates = [value + s for s in steps]
        elif isinstance(value, float):
            candidates = [round(value + s, 6) for s in (1.0, 0.5, 2.0)]
        elif isinstance(value, str):
            enum = properties.get(param, {}).get("enum")
            candidates = list(enum) if enum else list(donors.get(param, []))
        rng.shuffle(candidates)
        for candidate in candidates:
            if not value_is_acceptable(candidate, acceptable):
                new_args = dict(args)
                new_args[param] = candidate
                return new_args, f"{param}: {value!r} -> {candidate!r}"
    return None


def perturb_drop_required(args: Dict[str, Any], schema: dict, rng: random.Random) -> Optional[Tuple[Dict[str, Any], str]]:
    """P3: remove one required argument."""
    required = [p for p in schema["parameters"].get("required", []) if p in args]
    if not required:
        return None
    param = rng.choice(required)
    return {k: v for k, v in args.items() if k != param}, f"dropped {param}"


def perturb_extra_param(args: Dict[str, Any], schema: dict, rng: random.Random) -> Optional[Tuple[Dict[str, Any], str]]:
    """P4: add an argument that the schema does not define."""
    properties = schema["parameters"]["properties"]
    options = [(k, v) for k, v in EXTRA_PARAMS if k not in properties]
    if not options:
        return None
    param, value = rng.choice(options)
    new_args = dict(args)
    new_args[param] = value
    return new_args, f"added {param}={value!r}"


# ---
# DATASET ASSEMBLY
# ---

def load_category(raw_dir: Path, category: str) -> List[dict]:
    """Join BFCL questions with their ground truth for one category."""
    questions = load_jsonl(str(raw_dir / f"BFCL_v4_{category}.json"))
    answers = {r["id"]: r["ground_truth"] for r in load_jsonl(str(raw_dir / f"BFCL_v4_{category}_answer.json"))}
    items = []
    for q in questions:
        gt_list = answers[q["id"]]
        if len(gt_list) != 1 or len(q["question"]) != 1 or len(q["question"][0]) != 1:
            continue
        gt = gt_list[0]
        fname = next(iter(gt))
        schema = next((f for f in q["function"] if f["name"] == fname), None)
        if schema is None:
            continue
        items.append({
            "id": q["id"], "category": category,
            "question": q["question"][0][0]["content"],
            "functions": q["function"], "fname": fname, "gt": gt, "schema": schema,
        })
    return items


def collect_string_donors(items: List[dict]) -> Dict[str, List[str]]:
    """Collect string values seen for each parameter name, as replacement candidates for P1."""
    donors = defaultdict(set)
    for item in items:
        for param, acceptable in item["gt"][item["fname"]].items():
            for value in acceptable:
                if isinstance(value, str) and value:
                    donors[param].add(value)
    return {k: sorted(v) for k, v in donors.items()}


def build_pairs(raw_dir: Path, seed: int = 42) -> Tuple[List[dict], Counter]:
    """Build the full pair dataset and return it with construction statistics."""
    items = [it for cat in BFCL_CATEGORIES for it in load_category(raw_dir, cat)]
    donors = collect_string_donors(items)
    stats = Counter()
    records = []

    for item in items:
        fname, gt, schema = item["fname"], item["gt"], item["schema"]
        correct_args = build_correct_args(gt[fname], schema)
        if not is_valid_call(fname, correct_args, gt, schema):
            stats["skipped_correct_invalid"] += 1
            continue

        incorrect = []
        perturbations = [
            ("P1_wrong_value", lambda r: perturb_wrong_value(correct_args, gt[fname], schema, donors, r)),
            ("P3_drop_required", lambda r: perturb_drop_required(correct_args, schema, r)),
            ("P4_extra_param", lambda r: perturb_extra_param(correct_args, schema, r)),
        ]
        for ptype, fn in perturbations:
            result = fn(random.Random(f"{seed}:{item['id']}:{ptype}"))
            if result is None:
                stats[f"{ptype}_not_applicable"] += 1
                continue
            new_args, detail = result
            if is_valid_call(fname, new_args, gt, schema):
                stats[f"{ptype}_still_valid"] += 1
                continue
            incorrect.append({"type": ptype, "call": render_call(fname, new_args), "detail": detail})
            stats[ptype] += 1

        if not incorrect:
            stats["skipped_no_perturbation"] += 1
            continue
        records.append({
            "id": item["id"], "dataset": f"bfcl_{item['category']}", "bfcl_commit": BFCL_COMMIT,
            "question": item["question"], "functions": item["functions"],
            "fname": fname, "n_functions": len(item["functions"]),
            "correct_call": render_call(fname, correct_args), "incorrect": incorrect,
        })
        stats["items"] += 1
    return records, stats


def main():
    parser = argparse.ArgumentParser(description="Build matched tool-call pairs from BFCL")
    parser.add_argument("--raw-dir", default="dataset/tooldefer/raw", help="Directory with BFCL question and answer files")
    parser.add_argument("--output", "-o", default="dataset/tooldefer/bfcl_pairs.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    records, stats = build_pairs(Path(args.raw_dir), seed=args.seed)
    save_jsonl(records, args.output)
    print(f"Wrote {len(records)} items to {args.output}")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
