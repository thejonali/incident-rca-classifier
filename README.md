# Incident RCA Classifier

GRU/LSTM incident root-cause classifier with a workflow-routing demo on a synthetic RCA dataset.

This repository is an applied ML experiment, not an LLM agent. It trains recurrent neural-network classifiers that map incident descriptions to root-cause categories, then uses the predicted class to choose placeholder workflow instructions.

## What It Does

- Downloads the synthetic RCA incident training CSV on first training run.
- Builds a non-leaking incident text feature from operational fields.
- Trains GRU and LSTM classifiers on the same stratified train/validation/test split.
- Evaluates accuracy, macro F1, weighted F1, and per-class precision/recall/F1.
- Runs an interactive GRU workflow-routing demo for custom incidents.
- Runs a curated 10-sample batch demo that exercises mostly different classifications.

## Project Status

Experimental and portfolio-oriented. The tuned GRU performs strongly on this synthetic dataset, but the project is not production RCA tooling and should not be used for operational incident decisions without real incident data, stronger validation, monitoring, and human approval gates.

## Dataset

Source: [RCA Synthetic Training Dataset](https://www.kaggle.com/datasets/sakthivarshans/rca-synthetic-training-dataset)

Dataset license: Apache 2.0, as listed on Kaggle at the time this project was prepared.

Training automatically downloads `incidents_training_dataset.csv` at startup if it is missing locally:

```text
data/raw/incidents_training_dataset.csv
```

Dataset and generated files are intentionally ignored by Git:

- `data/raw/*.csv`
- `models/`
- `reports/`

Download only the dataset:

```bash
uv run python -m incident_data_classification.dataset
```

## Setup

Requirements:

- Python 3.11 through 3.13
- [uv](https://docs.astral.sh/uv/)

Install dependencies:

```bash
uv sync
```

## Train Models

Tuned runs used for the current reported scores:

```bash
uv run python -m incident_data_classification.train --model-type gru --max-rows 15000 --epochs 20 --batch-size 64 --vocab-size 12000 --max-length 128 --embedding-dim 96 --hidden-dim 96 --class-weights balanced --patience 5
uv run python -m incident_data_classification.train --model-type lstm --max-rows 15000 --epochs 12 --batch-size 64 --vocab-size 10000 --max-length 96 --embedding-dim 64 --hidden-dim 64 --learning-rate 0.003 --patience 4
```

Training restores the best validation checkpoint before test evaluation. The default monitor is `val_macro_f1`.

For a faster smoke run:

```bash
uv run python -m incident_data_classification.train --model-type gru --max-rows 500 --epochs 1
```

## Evaluate

```bash
uv run python -m incident_data_classification.evaluate
```

Current saved metrics are documented in [README_MODEL_SCORES.md](README_MODEL_SCORES.md).

## Run The Demo

Interactive GRU classifier:

```bash
uv run python -m incident_data_classification.interactive_gru
```

At the prompt:

- Enter an incident description to classify it.
- Enter `-1` to select one of 25 preconfigured natural-language issue descriptions.
- Enter `X` to exit.

Curated batch run:

```bash
uv run python -m incident_data_classification.interactive_gru --batch-count 10
```

The batch demo prints each message, expected class, predicted class, confidence, status, and workflow instructions.

Legacy dual-model prediction:

```bash
uv run python -m incident_data_classification.predict --text -1
```

## Architecture

```text
Dataset downloader
  -> CSV validation
  -> non-leaking text feature construction
  -> tokenizer + label encoder
  -> GRU/LSTM classifier training
  -> saved model artifacts
  -> evaluation reports
  -> GRU workflow-routing CLI
```

Model shape:

- `Embedding`
- `GRU` or `LSTM`
- `Dropout`
- linear classification head
- softmax at prediction time

The tuned GRU artifact used `embedding_dim=96`, `hidden_dim=96`, `dropout=0.25`, balanced class weights, and best-checkpoint restore.

## Validation

Local checks:

```bash
uv run pytest -q
uv run python -m incident_data_classification.evaluate
uv run python -m incident_data_classification.interactive_gru --batch-count 10
```

GitHub Actions runs `uv sync --locked` and `uv run pytest -q` on pushes, pull requests, and manual dispatch.

## Limitations

- The dataset is synthetic, so high scores can reflect synthetic generation patterns rather than real-world RCA performance.
- The model is a classifier, not an LLM, planner, or autonomous incident-response agent.
- Workflow instructions are placeholders that simulate routing. They do not execute remediation.
- The interactive natural-language samples are intentionally less model-aligned than the training data and may skew toward common classes.
- The LSTM clears 90% accuracy but has much weaker macro F1 than the GRU because it misses several smaller classes.
- No trained model binaries, raw data, or generated reports are committed.

## License

Licensed under the [Apache License 2.0](LICENSE).
