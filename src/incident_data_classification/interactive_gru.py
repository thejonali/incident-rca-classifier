from __future__ import annotations

import argparse
import json
import random
from importlib.resources import files
from pathlib import Path

from .config import DEFAULT_MODELS_DIR
from .predict import predict_one


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
    parser.add_argument("--text", type=str, default=None, help="Issue description. Use -1 for a random sample.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional deterministic sample selector.")
    parser.add_argument("--prefer-mps", action="store_true", help="Use Apple MPS if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text
    if text is None:
        text = input("Enter an issue description, or -1 for a preconfigured sample: ").strip()

    if text == "-1":
        sample = choose_sample(seed=args.sample_seed)
        text = sample["description"]
        print(f"Using sample: {sample['id']}")
        print(f"{text}\n")

    artifact_dir = args.models_dir / "gru"
    if not (artifact_dir / "model.pt").exists():
        raise FileNotFoundError(f"GRU model not found at {artifact_dir}. Train the model first.")

    result = predict_one(artifact_dir, text, prefer_mps=args.prefer_mps)
    print(f"Classification: {result['label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("Top alternatives:")
    for item in result["top3"][1:]:
        print(f"  - {item['label']}: {item['confidence']:.3f}")

    print("\nPretend workflow instructions:")
    for index, instruction in enumerate(get_workflow_instructions(result["label"]), start=1):
        print(f"{index}. {instruction}")


if __name__ == "__main__":
    main()
