from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    FEATURE_PROFILES,
    TARGET_COLUMN,
)
from .data import load_incidents, split_dataset
from .dataset import resolve_incidents_csv
from .retrieval import load_retrieval_index, retrieve_similar_incidents
from .train_baseline import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a retrieval quality review report")
    parser.add_argument("--csv", type=Path, default=None, help="Optional local CSV path. Defaults to data/raw.")
    parser.add_argument("--force-download", action="store_true", help="Redownload the local CSV before evaluation.")
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=DEFAULT_FEATURE_PROFILE,
        help="Input field profile used by the retrieval index.",
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = resolve_incidents_csv(args.csv, force_download=args.force_download)
    df = load_incidents(csv_path, max_rows=args.max_rows, feature_profile=args.feature_profile)
    splits = split_dataset(df)
    sample_df = splits.test_df.head(args.sample_size)
    index = load_retrieval_index(args.models_dir / "retrieval" / args.feature_profile / "index.joblib")

    review_cases = []
    unfiltered_top1_matches = 0
    unfiltered_topk_matches = 0
    filtered_similarities = []

    for _, row in sample_df.iterrows():
        expected_category = str(row[TARGET_COLUMN])
        query_text = str(row["input_text"])
        unfiltered_matches = retrieve_similar_incidents(index, query_text, top_k=args.top_k)
        filtered_matches = retrieve_similar_incidents(index, query_text, category=expected_category, top_k=args.top_k)

        unfiltered_categories = [match["root_cause_category"] for match in unfiltered_matches]
        if unfiltered_categories and unfiltered_categories[0] == expected_category:
            unfiltered_top1_matches += 1
        if expected_category in unfiltered_categories:
            unfiltered_topk_matches += 1
        if filtered_matches:
            filtered_similarities.append(float(filtered_matches[0]["similarity"]))

        review_cases.append(
            {
                "incident_id": row["incident_id"],
                "expected_category": expected_category,
                "query_text": query_text,
                "unfiltered_matches": unfiltered_matches,
                "category_filtered_matches": filtered_matches,
            }
        )

    report = {
        "benchmark": "retrieval_review_sample",
        "feature_profile": args.feature_profile,
        "sample_size": int(len(sample_df)),
        "top_k": args.top_k,
        "unfiltered_category_match_at_1": unfiltered_top1_matches / max(1, len(sample_df)),
        "unfiltered_category_match_at_k": unfiltered_topk_matches / max(1, len(sample_df)),
        "category_filtered_mean_top1_similarity": float(np.mean(filtered_similarities)) if filtered_similarities else 0.0,
        "review_cases": review_cases,
    }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.reports_dir / f"retrieval_{args.feature_profile}_review.json"
    save_json(report_path, report)
    print(f"Saved retrieval review report to {report_path}")
    print(f"Unfiltered category match@1: {report['unfiltered_category_match_at_1']:.3f}")
    print(f"Unfiltered category match@{args.top_k}: {report['unfiltered_category_match_at_k']:.3f}")
    print(f"Filtered mean top-1 similarity: {report['category_filtered_mean_top1_similarity']:.3f}")


if __name__ == "__main__":
    main()
