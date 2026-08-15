from __future__ import annotations

import argparse
import json
import random
from importlib.resources import files
from pathlib import Path

from .config import DEFAULT_MODELS_DIR
from .predict import predict_one


PROMPT = "Enter an issue description, -1 for a preconfigured sample, or X to exit: "
MAX_BATCH_MESSAGE_LENGTH = 360

WORKFLOW_INSTRUCTIONS = {
    "CASCADING_FAILURE": [
        "Start dependency-impact workflow and identify the first failing upstream service.",
        "Throttle retries and enable circuit breakers on affected callers.",
        "Page owners for both the origin service and the highest-traffic dependent service.",
    ],
    "CONFIGURATION_ERROR": [
        "Start configuration rollback workflow for the affected service.",
        "Compare current runtime config against the last known good deployment.",
        "Validate secrets, environment variables, feature flags, and service endpoints.",
    ],
    "DATA_CORRUPTION": [
        "Start data-integrity workflow and freeze risky write paths.",
        "Capture affected record IDs and compare them against the latest clean backup.",
        "Prepare a repair or replay plan before resuming automated writes.",
    ],
    "DEPENDENCY_FAILURE": [
        "Start dependency-failure workflow and verify health of upstream systems.",
        "Enable fallback behavior or degrade noncritical features.",
        "Reduce retry pressure and monitor queue depth until the dependency recovers.",
    ],
    "DEPLOYMENT_REGRESSION": [
        "Start deployment rollback workflow for the latest release.",
        "Compare error rates between current and previous versions.",
        "Pause progressive rollout and run post-rollback smoke checks.",
    ],
    "INFRASTRUCTURE_FAILURE": [
        "Start infrastructure-recovery workflow and inspect node, zone, and cluster health.",
        "Reschedule workloads away from unhealthy capacity.",
        "Verify replica counts, pod disruption budgets, and load balancer targets.",
    ],
    "RESOURCE_EXHAUSTION": [
        "Start resource-scaling workflow for CPU, memory, disk, and I/O saturation.",
        "Apply a short-term capacity increase or reduce workload pressure.",
        "Add alerts around the saturated resource and review autoscaling thresholds.",
    ],
    "RESOURCE_LEAK": [
        "Start leak-containment workflow and capture heap, file descriptor, or connection metrics.",
        "Restart affected instances only after collecting diagnostic snapshots.",
        "Open a fix task for the leaking code path and add regression detection.",
    ],
    "SCHEDULED_JOB_FAILURE": [
        "Start scheduled-job recovery workflow and inspect scheduler retries.",
        "Rerun failed jobs only after confirming idempotency.",
        "Notify downstream report or pipeline owners about stale outputs.",
    ],
    "SECURITY_INCIDENT": [
        "Start security triage workflow and preserve relevant logs.",
        "Apply temporary access controls, rate limits, or account protections.",
        "Escalate to the security owner for scope assessment and evidence review.",
    ],
    "THIRD_PARTY_FAILURE": [
        "Start third-party degradation workflow and confirm provider status.",
        "Switch to fallback provider or queue noncritical requests.",
        "Track provider error rate, rate limits, and recovery ETA.",
    ],
    "TRAFFIC_OVERLOAD": [
        "Start traffic-surge workflow and inspect request distribution.",
        "Scale entry-point services and apply rate limits to abusive or noncritical traffic.",
        "Coordinate with product or marketing owners about expected demand.",
    ],
}


def load_sample_descriptions() -> list[dict[str, str]]:
    sample_path = files("incident_data_classification").joinpath("samples/issue_descriptions.json")
    return json.loads(sample_path.read_text(encoding="utf-8"))


def load_batch_descriptions() -> list[dict[str, str]]:
    sample_path = files("incident_data_classification").joinpath("samples/batch_issue_descriptions.json")
    return json.loads(sample_path.read_text(encoding="utf-8"))


def build_reference_batch(reference_rows: list[dict], count: int) -> list[dict[str, str]]:
    samples = []
    seen_labels = set()
    for row in reference_rows:
        label = row.get("root_cause_category")
        text = row.get("input_text")
        if not label or not text or label in seen_labels:
            continue
        seen_labels.add(label)
        samples.append(
            {
                "id": f"reference-{label.lower()}",
                "expected_classification": label,
                "description": text,
            }
        )
        if len(samples) >= count:
            break
    return samples


def load_reference_batch(artifact_dir: Path, count: int) -> list[dict[str, str]]:
    reference_path = artifact_dir / "reference_incidents.json"
    if not reference_path.exists():
        return []
    rows = json.loads(reference_path.read_text(encoding="utf-8"))
    return build_reference_batch(rows, count)


def choose_sample(seed: int | None = None) -> dict[str, str]:
    samples = load_sample_descriptions()
    rng = random.Random(seed)
    return rng.choice(samples)


def get_workflow_instructions(label: str) -> list[str]:
    return WORKFLOW_INSTRUCTIONS.get(
        label,
        [
            "Start general incident triage workflow.",
            "Capture impact, timeline, metrics, logs, and recent changes.",
            "Assign an owner to validate the classification before automated remediation.",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive GRU incident classifier")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--text", type=str, default=None, help="Issue description. Use -1 for a random sample, X to exit.")
    parser.add_argument("--batch-count", type=int, default=0, help="Run this many curated batch samples and exit.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional deterministic sample selector.")
    parser.add_argument("--prefer-mps", action="store_true", help="Use Apple MPS if available")
    return parser.parse_args()


def resolve_input_text(text: str, sample_seed: int | None = None) -> str:
    if text == "-1":
        sample = choose_sample(seed=sample_seed)
        text = sample["description"]
        print(f"Using sample: {sample['id']}")
        print(f"{text}\n")
    return text


def print_prediction(result: dict) -> None:
    print(f"Classification: {result['label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("Top alternatives:")
    for item in result["top3"][1:]:
        print(f"  - {item['label']}: {item['confidence']:.3f}")

    print("\nWorkflow Instructions:")
    for index, instruction in enumerate(get_workflow_instructions(result["label"]), start=1):
        print(f"{index}. {instruction}")
    print()


def print_batch_prediction(sample: dict[str, str], result: dict) -> None:
    expected = sample.get("expected_classification", "-")
    message = sample.get("description", "")
    if len(message) > MAX_BATCH_MESSAGE_LENGTH:
        message = f"{message[:MAX_BATCH_MESSAGE_LENGTH].rstrip()}..."
    status = "ok" if expected == result["label"] else "check"
    print(f"Sample: {sample['id']}")
    print(f"Message: {message}")
    print(f"Expected: {expected}")
    print(f"Classification: {result['label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Status: {status}")
    print("Workflow Instructions:")
    for index, instruction in enumerate(get_workflow_instructions(result["label"]), start=1):
        print(f"{index}. {instruction}")
    print()


def run_batch(count: int, artifact_dir: Path, prefer_mps: bool = False) -> None:
    if count <= 0:
        return
    if not (artifact_dir / "model.pt").exists():
        raise FileNotFoundError(f"GRU model not found at {artifact_dir}. Train the model first.")

    samples = load_reference_batch(artifact_dir, count)
    if len(samples) < count:
        samples = load_batch_descriptions()[:count]

    counts: dict[str, int] = {}
    for sample in samples:
        result = predict_one(artifact_dir, sample["description"], prefer_mps=prefer_mps)
        counts[result["label"]] = counts.get(result["label"], 0) + 1
        print_batch_prediction(sample, result)

    print("Batch classification counts:")
    for label, total in sorted(counts.items()):
        print(f"- {label}: {total}")


def run_once(text: str, artifact_dir: Path, sample_seed: int | None = None, prefer_mps: bool = False) -> None:
    resolved_text = resolve_input_text(text, sample_seed=sample_seed)
    if not resolved_text:
        return
    if not (artifact_dir / "model.pt").exists():
        raise FileNotFoundError(f"GRU model not found at {artifact_dir}. Train the model first.")

    result = predict_one(artifact_dir, resolved_text, prefer_mps=prefer_mps)
    print_prediction(result)


def main() -> None:
    args = parse_args()
    artifact_dir = args.models_dir / "gru"
    if args.batch_count:
        run_batch(args.batch_count, artifact_dir, prefer_mps=args.prefer_mps)
        return

    if args.text is not None:
        if args.text.strip().upper() == "X":
            return
        run_once(args.text.strip(), artifact_dir, sample_seed=args.sample_seed, prefer_mps=args.prefer_mps)
        return

    print("Enter X to exit.")
    while True:
        text = input(PROMPT).strip()
        if text.upper() == "X":
            break
        run_once(text, artifact_dir, sample_seed=args.sample_seed, prefer_mps=args.prefer_mps)


if __name__ == "__main__":
    main()
