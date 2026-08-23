from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import DEFAULT_FEATURE_PROFILE, DEFAULT_MODELS_DIR, FEATURE_PROFILES
from .data import normalize_text
from .labels import LabelEncoder
from .model import RecurrentIncidentClassifier, get_device
from .tokenizer import TextTokenizer


SAMPLE_TEXT = (
    "payment-service in production is showing disk io saturation and rising latency "
    "after the nightly analytics job started. threshold alerts fired, customers are "
    "seeing slow checkout responses, and cpu is also elevated."
)


def resolve_artifact_dir(models_dir: Path, model_name: str, feature_profile: str | None = None) -> Path:
    if feature_profile is None:
        default_profile_dir = models_dir / model_name / DEFAULT_FEATURE_PROFILE
        if default_profile_dir.exists():
            return default_profile_dir
        return models_dir / model_name

    profile_dir = models_dir / model_name / feature_profile
    return profile_dir


def load_model(artifact_dir: Path, prefer_mps: bool) -> tuple[RecurrentIncidentClassifier, TextTokenizer, LabelEncoder, torch.device]:
    checkpoint = torch.load(artifact_dir / "model.pt", map_location="cpu")
    tokenizer = TextTokenizer.load(artifact_dir / "tokenizer.json")
    label_encoder = LabelEncoder.load(artifact_dir / "label_encoder.json")
    device = get_device(prefer_mps=prefer_mps)
    model = RecurrentIncidentClassifier(
        model_type=checkpoint["model_type"],
        vocab_size=checkpoint["vocab_size"],
        num_classes=checkpoint["num_classes"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, label_encoder, device


def predict_one(artifact_dir: Path, text: str, prefer_mps: bool = False) -> dict:
    model, tokenizer, label_encoder, device = load_model(artifact_dir, prefer_mps=prefer_mps)
    input_ids = torch.tensor([tokenizer.encode(normalize_text(text))], dtype=torch.long).to(device)
    with torch.no_grad():
        probabilities = F.softmax(model(input_ids), dim=1).cpu()[0]

    top_probability, top_index = torch.max(probabilities, dim=0)
    labels_by_id = label_encoder.id_to_label
    top_label = labels_by_id[int(top_index.item())]
    top3 = torch.topk(probabilities, k=min(3, len(probabilities)))

    return {
        "model": artifact_dir.name,
        "label": top_label,
        "confidence": float(top_probability.item()),
        "top3": [
            {
                "label": labels_by_id[int(index.item())],
                "confidence": float(probability.item()),
            }
            for probability, index in zip(top3.values, top3.indices, strict=True)
        ],
        "reference": find_reference(artifact_dir, top_label),
    }


def find_reference(artifact_dir: Path, label: str) -> dict | None:
    reference_path = artifact_dir / "reference_incidents.json"
    if not reference_path.exists():
        return None
    rows = json.loads(reference_path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("root_cause_category") == label:
            return {
                "incident_id": row.get("incident_id"),
                "remediation_that_worked": row.get("remediation_that_worked"),
                "prevention_recommendation": row.get("prevention_recommendation"),
            }
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRU and LSTM incident predictions")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=None,
        help="Feature profile artifact to load.",
    )
    parser.add_argument("--text", type=str, default=None, help="Incident text. Use -1 for built-in sample.")
    parser.add_argument("--prefer-mps", action="store_true", help="Use Apple MPS if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text
    if text is None:
        text = input("Enter incident description, or -1 for sample incident: ").strip()
    if text == "-1":
        text = SAMPLE_TEXT
        print(f"Using sample incident:\n{text}\n")

    for model_name in ["gru", "lstm"]:
        artifact_dir = resolve_artifact_dir(args.models_dir, model_name, args.feature_profile)
        if not (artifact_dir / "model.pt").exists():
            print(f"{model_name.upper()} model not found at {artifact_dir}; train it first.")
            continue

        result = predict_one(artifact_dir, text, prefer_mps=args.prefer_mps)
        print(f"{model_name.upper()}: {result['label']} ({result['confidence']:.3f})")
        print("Top classes:")
        for item in result["top3"]:
            print(f"  - {item['label']}: {item['confidence']:.3f}")
        if result["reference"]:
            print(f"Reference incident: {result['reference']['incident_id']}")
            print(f"Remediation: {result['reference']['remediation_that_worked']}")
            print(f"Prevention: {result['reference']['prevention_recommendation']}")
        print()


if __name__ == "__main__":
    main()
