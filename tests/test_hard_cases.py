from pathlib import Path

from incident_data_classification.config import DEFAULT_HARD_CASES_PATH, ROOT_CAUSE_CATEGORIES
from incident_data_classification.evaluate_hard_cases import build_hard_case_report, load_hard_cases


def test_hard_cases_are_valid_and_separate_from_training_data():
    cases = load_hard_cases(DEFAULT_HARD_CASES_PATH)
    ids = [case["id"] for case in cases]
    categories = {case["expected_category"] for case in cases}

    assert len(cases) >= 100
    assert len(ids) == len(set(ids))
    assert categories <= set(ROOT_CAUSE_CATEGORIES)
    assert len(categories) == len(ROOT_CAUSE_CATEGORIES)
    assert DEFAULT_HARD_CASES_PATH.parent.name == "evaluation"
    assert DEFAULT_HARD_CASES_PATH.parent.parent.name == "data"


def test_hard_case_report_is_separate_benchmark():
    cases = [
        {
            "id": "HARD-T1",
            "input": "checkout is slow",
            "expected_category": "TRAFFIC_OVERLOAD",
            "difficulty": "hard",
            "scenario_type": "ambiguous",
            "notes": "test case",
        },
        {
            "id": "HARD-T2",
            "input": "provider is down",
            "expected_category": "THIRD_PARTY_FAILURE",
            "difficulty": "hard",
            "scenario_type": "dependency",
            "notes": "test case",
        },
    ]

    report = build_hard_case_report(
        cases=cases,
        predictions=["TRAFFIC_OVERLOAD", "DEPENDENCY_FAILURE"],
        model_name="linear_svm",
        feature_profile="alert_only",
        inference_seconds=0.002,
    )

    assert report["benchmark"] == "hard_evaluation_set"
    assert report["case_count"] == 2
    assert report["test_accuracy"] == 0.5
    assert len(report["failures"]) == 1
