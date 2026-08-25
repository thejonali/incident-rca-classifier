from incident_data_classification.transformer_utils import (
    TransformerIncidentDataset,
    directory_size_mb,
    resolve_transformer_artifact_dir,
)


class DummyTokenizer:
    def __call__(self, texts, truncation, padding, max_length):
        assert truncation is True
        assert padding is True
        assert max_length == 8
        return {
            "input_ids": [[1, 2, 0], [1, 3, 0]][: len(texts)],
            "attention_mask": [[1, 1, 0], [1, 1, 0]][: len(texts)],
        }


def test_transformer_incident_dataset_includes_labels_when_present():
    dataset = TransformerIncidentDataset(["one", "two"], [0, 1], DummyTokenizer(), max_length=8)

    item = dataset[1]

    assert len(dataset) == 2
    assert item["input_ids"].tolist() == [1, 3, 0]
    assert item["attention_mask"].tolist() == [1, 1, 0]
    assert item["labels"].item() == 1


def test_transformer_incident_dataset_supports_unlabeled_prediction():
    dataset = TransformerIncidentDataset(["one"], None, DummyTokenizer(), max_length=8)

    assert "labels" not in dataset[0]


def test_resolve_transformer_artifact_dir_uses_model_family_folder(tmp_path):
    path = resolve_transformer_artifact_dir(tmp_path, "distilbert", "alert_only")

    assert path == tmp_path / "transformers" / "distilbert" / "alert_only"


def test_directory_size_mb_counts_files(tmp_path):
    path = tmp_path / "artifact"
    path.mkdir()
    (path / "weights.bin").write_bytes(b"0" * 1024)

    assert directory_size_mb(path) > 0
