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

These tuned runs are still practical on a 16 GB MacBook, but the GRU run takes a few minutes on CPU:

```bash
uv run python -m incident_data_classification.train --model-type gru --max-rows 15000 --epochs 20 --batch-size 64 --vocab-size 12000 --max-length 128 --embedding-dim 96 --hidden-dim 96 --class-weights balanced --patience 5
uv run python -m incident_data_classification.train --model-type lstm --max-rows 15000 --epochs 12 --batch-size 64 --vocab-size 10000 --max-length 96 --embedding-dim 64 --hidden-dim 64 --learning-rate 0.003 --patience 4
```

Training restores the best validation checkpoint before evaluating on the test split. By default, early stopping watches `val_macro_f1` with `--patience 3`.

## Predict

Run the interactive GRU-only workflow demo:

```bash
uv run python -m incident_data_classification.interactive_gru
```

Enter `-1` at the prompt to select one of 25 preconfigured sample issue descriptions.
Enter `X` to exit the prompt loop.

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
