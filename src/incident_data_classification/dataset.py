from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import DATASET_DOWNLOAD_URL, DEFAULT_CSV_PATH


def download_incidents_csv(
    csv_path: Path = DEFAULT_CSV_PATH,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """Download the incidents CSV to the local raw-data folder."""
    if csv_path.exists() and not force:
        return csv_path

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATASET_DOWNLOAD_URL, headers={"User-Agent": "incident-rca-classifier/0.1"})

    try:
        with urlopen(request, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(delete=False, dir=csv_path.parent) as temp_file:
                shutil.copyfileobj(response, temp_file)
                temp_path = Path(temp_file.name)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Unable to download incidents_training_dataset.csv from the public Kaggle endpoint. "
            "Check network access or pass --csv with a local file path."
        ) from exc

    temp_path.replace(csv_path)
    return csv_path


def resolve_incidents_csv(csv_path: Path | None = None, force_download: bool = False) -> Path:
    if csv_path is not None:
        return csv_path
    return download_incidents_csv(force=force_download)


def main() -> None:
    path = download_incidents_csv(force=True)
    print(path)


if __name__ == "__main__":
    main()
