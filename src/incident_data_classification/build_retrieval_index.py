from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_FEATURE_PROFILE, DEFAULT_MODELS_DIR, FEATURE_PROFILES
from .data import load_incidents, split_dataset
from .dataset import resolve_incidents_csv
from .retrieval import build_retrieval_index, save_retrieval_index
from .train_baseline import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local similar-incident retrieval index")
    parser.add_argument("--csv", type=Path, default=None, help="Optional local CSV path. Defaults to data/raw.")
    parser.add_argument("--force-download", action="store_true", help="Redownload the local CSV before indexing.")
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=DEFAULT_FEATURE_PROFILE,
        help="Input field profile used to build indexed incident text.",
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument(
        "--source-split",
        choices=("train", "all"),
        default="train",
        help="Index only the training split by default to avoid test-set retrieval leakage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = resolve_incidents_csv(args.csv, force_download=args.force_download)
    df = load_incidents(csv_path, max_rows=args.max_rows, feature_profile=args.feature_profile)
    source_df = split_dataset(df).train_df if args.source_split == "train" else df
    index = build_retrieval_index(source_df, feature_profile=args.feature_profile)

    artifact_dir = args.models_dir / "retrieval" / args.feature_profile
    save_retrieval_index(artifact_dir / "index.joblib", index)
    save_json(
        artifact_dir / "metadata.json",
        {
            "feature_profile": args.feature_profile,
            "source_split": args.source_split,
            "rows_used": int(len(source_df)),
            "max_rows": args.max_rows,
            "embedding_type": "tfidf_1_2_gram",
            "similarity": "cosine",
            "category_filtering": "filter_by_predicted_or_expected_root_cause_category",
        },
    )

    print(f"Saved retrieval index to {artifact_dir / 'index.joblib'}")
    print(f"Indexed incidents: {len(source_df)}")


if __name__ == "__main__":
    main()
