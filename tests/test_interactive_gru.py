from incident_data_classification.interactive_gru import (
    build_reference_batch,
    get_workflow_instructions,
    load_batch_descriptions,
    load_sample_descriptions,
    print_batch_prediction,
    print_prediction,
    resolve_input_text,
)


def test_loads_25_preconfigured_issue_descriptions():
    samples = load_sample_descriptions()

    assert len(samples) == 25
    assert all(sample["id"] and sample["description"] for sample in samples)


def test_loads_curated_batch_descriptions():
    samples = load_batch_descriptions()
    expected = {sample["expected_classification"] for sample in samples}

    assert len(samples) == 10
    assert len(expected) >= 8
    assert all(sample["id"] and sample["description"] for sample in samples)


def test_build_reference_batch_selects_one_sample_per_class():
    rows = [
        {"root_cause_category": "A", "input_text": "first a"},
        {"root_cause_category": "A", "input_text": "second a"},
        {"root_cause_category": "B", "input_text": "first b"},
    ]

    samples = build_reference_batch(rows, count=2)

    assert [sample["expected_classification"] for sample in samples] == ["A", "B"]
    assert [sample["description"] for sample in samples] == ["first a", "first b"]


def test_workflow_instructions_exist_for_known_classification():
    instructions = get_workflow_instructions("RESOURCE_EXHAUSTION")

    assert len(instructions) >= 3
    assert "resource" in " ".join(instructions).lower()


def test_workflow_instructions_fallback_for_unknown_classification():
    instructions = get_workflow_instructions("UNKNOWN")

    assert instructions[0] == "Start general incident triage workflow."


def test_resolve_input_text_uses_seeded_sample():
    text = resolve_input_text("-1", sample_seed=7)

    assert "login-service" in text


def test_print_prediction_uses_workflow_instructions_heading_and_trailing_newline(capsys):
    print_prediction(
        {
            "label": "RESOURCE_EXHAUSTION",
            "confidence": 0.9876,
            "top3": [
                {"label": "RESOURCE_EXHAUSTION", "confidence": 0.9876},
                {"label": "DEPENDENCY_FAILURE", "confidence": 0.01},
            ],
        }
    )

    output = capsys.readouterr().out

    assert "Workflow Instructions:" in output
    assert "Pretend workflow instructions:" not in output
    assert output.endswith("\n\n")


def test_print_batch_prediction_includes_expected_and_status(capsys):
    print_batch_prediction(
        {"id": "sample-a", "expected_classification": "RESOURCE_EXHAUSTION"},
        {
            "label": "RESOURCE_EXHAUSTION",
            "confidence": 0.91,
            "top3": [{"label": "RESOURCE_EXHAUSTION", "confidence": 0.91}],
        },
    )

    output = capsys.readouterr().out

    assert "Sample: sample-a" in output
    assert "Expected: RESOURCE_EXHAUSTION" in output
    assert "Status: ok" in output
