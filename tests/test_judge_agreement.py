"""Sampling and agreement math for judge validation."""

from src.eval.judge_agreement import agreement, pick_sample

ROWS = [{"id": f"ok{i}", "outcome": "correct"} for i in range(10)] + [
    {"id": "bad1", "outcome": "unsafe_error"}, {"id": "bad2", "outcome": "flagged_error"}]


def test_sample_puts_errors_first_and_is_reproducible():
    sample = pick_sample(ROWS, 5)
    assert sample[:2] == ["bad1", "bad2"] and len(sample) == 5
    assert pick_sample(ROWS, 5) == sample


def test_agreement_counts_matches_and_lists_disagreements():
    human = {"a": {"faithfulness": "supported", "correctness": "correct"},
             "b": {"faithfulness": "supported", "correctness": "incorrect"}}
    judge = {"a": {"faithfulness": "supported", "correctness": "correct"},
             "b": {"faithfulness": "supported", "correctness": "partially_correct"}}
    result = agreement(human, judge)
    assert result["faithfulness"]["agreement"] == 1.0
    assert result["correctness"] == {"agreement": 0.5, "disagreements": [("b", "incorrect", "partially_correct")]}
