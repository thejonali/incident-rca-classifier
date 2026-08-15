from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_CSV_PATH, DEFAULT_DATASET_CACHE_DIR, KAGGLE_DATASET_FILE, KAGGLE_DATASET_REF


def download_incidents_csv(
    cache_dir: Path = DEFAULT_DATASET_CACHE_DIR,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Download the incidents CSV with the official Kaggle Python package."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / KAGGLE_DATASET_FILE
    if destination.exists() and not force:
        return destination

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Install project dependencies first with `uv sync`.") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Run `uv run kaggle auth login`, or configure "
            "`KAGGLE_USERNAME` and `KAGGLE_KEY` from your Kaggle API token."
        ) from exc

    api.dataset_download_file(
        dataset=KAGGLE_DATASET_REF,
        file_name=KAGGLE_DATASET_FILE,
        path=str(cache_dir),
        force=force,
        quiet=quiet,
    )

    if not destination.exists():
        raise FileNotFoundError(f"Kaggle download completed, but {destination} was not created.")
    return destination


def resolve_incidents_csv(csv_path: Path | None = None, force_download: bool = False) -> Path:
    if csv_path is not None:
        return csv_path
    if DEFAULT_CSV_PATH.exists() and not force_download:
        return DEFAULT_CSV_PATH
    return download_incidents_csv(force=force_download)


def main() -> None:
    path = download_incidents_csv()
    print(path)


if __name__ == "__main__":
    main()

