"""Tests for the ToolDefer pair builder, prompts and analysis."""

import sys
from pathlib import Path
from statistics import NormalDist

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from build_bfcl_pairs import build_correct_args, build_pairs, is_valid_call, value_is_acceptable
import numpy as np

from tooldefer_analyze import auc, dds, sdt_measures
from tooldefer_prompts import ANSWER_ORDERS, CONDITIONS, answer_tokens, build_prompt

RAW_DIR = Path(__file__).parent.parent.parent.parent / "dataset" / "tooldefer" / "raw"

SCHEMA = {
    "name": "area",
    "parameters": {
        "type": "dict",
        "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}, "unit": {"type": "string"}},
        "required": ["base", "height"],
    },
}
GT = {"area": {"base": [10], "height": [5], "unit": ["units", ""]}}


def test_correct_call_omits_optional_and_is_valid():
    args = build_correct_args(GT["area"], SCHEMA)
    assert args == {"base": 10, "height": 5}
    assert is_valid_call("area", args, GT, SCHEMA)


@pytest.mark.parametrize("args", [
    {"base": 11, "height": 5},
    {"base": 10},
    {"base": 10, "height": 5, "verbose": True},
    {"base": 10, "height": 5, "unit": "metres"},
])
def test_invalid_calls_are_rejected(args):
    assert not is_valid_call("area", args, GT, SCHEMA)


def test_wrong_function_is_rejected():
    assert not is_valid_call("volume", {"base": 10, "height": 5}, GT, SCHEMA)


def test_string_matching_follows_bfcl_normalisation():
    assert value_is_acceptable("new york", ["New York"])
    assert not value_is_acceptable("Boston", ["New York"])
    assert not value_is_acceptable(1, [True])


@pytest.mark.skipif(not RAW_DIR.exists(), reason="BFCL raw files not downloaded")
def test_built_dataset_has_distinct_correct_and_incorrect_calls():
    records, stats = build_pairs(RAW_DIR)
    assert len(records) > 500
    assert not any(key.endswith("_still_valid") for key in stats)
    for record in records:
        calls = [inc["call"] for inc in record["incorrect"]]
        assert record["correct_call"] not in calls
        assert len(set(calls)) == len(calls)


def test_prompts_differ_only_in_wrapper():
    functions, question, call = [SCHEMA], "Area of a triangle with base 10 and height 5?", "area(base=10, height=5)"
    for condition in CONDITIONS:
        for order in ANSWER_ORDERS:
            for answer_format in ("numbered", "yesno"):
                prompt = build_prompt(condition, question, functions, call, order, answer_format)
                assert call in prompt and question in prompt and '"name": "area"' in prompt
    assert "Human Engineer: [function call]" in build_prompt("c2_human", question, functions, call, "yes_first")
    assert '{"status": "success"}' in build_prompt("c3", question, functions, call, "yes_first")
    assert '{"status": "success"}' not in build_prompt("c2", question, functions, call, "yes_first")


def test_answer_tokens_follow_order():
    assert answer_tokens("yes_first") == ("1", "2")
    assert answer_tokens("no_first") == ("2", "1")
    assert answer_tokens("no_first", "yesno") == ("Yes", "No")


def test_sdt_matches_hand_computation():
    z = NormalDist().inv_cdf
    d_prime, criterion = sdt_measures(80, 100, 20, 100)
    hit, fa = 80.5 / 101, 20.5 / 101
    assert d_prime == pytest.approx(z(hit) - z(fa))
    assert criterion == pytest.approx(-0.5 * (z(hit) + z(fa)))
    assert criterion == pytest.approx(0.0, abs=1e-9)


def test_lenient_overseer_has_negative_criterion():
    _, criterion = sdt_measures(95, 100, 60, 100)
    assert criterion < 0


def test_dds_equals_twice_the_approval_rate_shift():
    hit_c1, fa_c1, hit_c2, fa_c2 = 0.70, 0.30, 0.82, 0.45
    approval_shift = ((hit_c2 + fa_c2) - (hit_c1 + fa_c1)) / 2
    assert dds(hit_c1, fa_c1, hit_c2, fa_c2) == pytest.approx(200 * approval_shift)


def test_auc_ranges_from_chance_to_perfect():
    assert auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5
    assert auc(np.array([0.9, 0.1]), np.array([0.5])) == 0.5
