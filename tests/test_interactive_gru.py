from incident_data_classification.interactive_gru import (
    get_workflow_instructions,
    load_sample_descriptions,
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
