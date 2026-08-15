from incident_data_classification.interactive_gru import (
    get_workflow_instructions,
    load_sample_descriptions,
    print_prediction,
    resolve_input_text,
)


def test_loads_25_preconfigured_issue_descriptions():
    samples = load_sample_descriptions()

    assert len(samples) == 25
    assert all(sample["id"] and sample["description"] for sample in samples)


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
