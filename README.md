# Incident Data Classification

Small GRU and LSTM experiments for classifying incident root-cause categories from the Kaggle RCA synthetic training dataset.

## Setup

```bash
uv sync
```

## Dataset

Training automatically downloads `incidents_training_dataset.csv` at startup if it is missing locally.

```text
data/raw/incidents_training_dataset.csv
```

To download only the dataset without training:

```bash
uv run python -m incident_data_classification.dataset
```

## Train Small Models

These defaults are intentionally small for a MacBook-class machine:

```bash
uv run python -m incident_data_classification.train --model-type gru --max-rows 3000 --epochs 8
uv run python -m incident_data_classification.train --model-type lstm --max-rows 3000 --epochs 8
```

Training restores the best validation checkpoint before evaluating on the test split. By default, early stopping watches `val_macro_f1` with `--patience 3`.

## Predict

Use `-1` for the built-in sample incident:

```bash
uv run python -m incident_data_classification.predict --text -1
```

Or pass your own text:

```bash
uv run python -m incident_data_classification.predict --text "payment-service has rising latency and disk io saturation after the analytics batch job started"
```

## Compare Saved Metrics

```bash
uv run python -m incident_data_classification.evaluate
```
