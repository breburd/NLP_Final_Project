from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import bert_baseline


class FakeTokenizer:
    sep_token = "[SEP]"

    def encode_plus(
        self,
        text,
        add_special_tokens=True,
        max_length=10,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    ):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        }

    def __call__(self, prompt, return_tensors="pt", truncation=True):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, values, skip_special_tokens=True):
        return "simple explanation"


class FakeExplanationModel:
    device = "cpu"

    def generate(self, **kwargs):
        return torch.tensor([[10, 11, 12]])


def test_enron_dataset_len():
    dataset = bert_baseline.EnronDataset(
        from_user=["a"],
        to=["b"],
        subject=["hello"],
        email=["body"],
        privileged=[1],
        tokenizer=FakeTokenizer(),
        max_len=12,
    )

    assert len(dataset) == 1


def test_enron_dataset_getitem_returns_dictionary():
    dataset = bert_baseline.EnronDataset(
        from_user=["a"],
        to=["b"],
        subject=["hello"],
        email=["body"],
        privileged=[1],
        tokenizer=FakeTokenizer(),
        max_len=12,
    )

    item = dataset[0]

    assert "input_ids" in item
    assert "attention_mask" in item
    assert item["labels"].item() == 1
    assert "Is this email privileged?" in item["text"]


def test_compute_loss_returns_tensor():
    trainer = bert_baseline.MyTrainer.__new__(bert_baseline.MyTrainer)
    trainer.class_weights = None

    class FakeModel:
        config = SimpleNamespace(num_labels=2)

        def __call__(self, **inputs):
            return {"logits": torch.tensor([[2.0, 1.0]], dtype=torch.float32)}

    inputs = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([0])}
    loss = trainer.compute_loss(FakeModel(), inputs)

    assert isinstance(loss, torch.Tensor)


def test_maybe_take_some_rows_returns_smaller_dataframe():
    df = pd.DataFrame({"value": [1, 2, 3, 4]})

    result = bert_baseline.maybe_take_some_rows(df, 2)

    assert len(result) == 2


def test_metric_function_returns_basic_scores():
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0])

    result = bert_baseline.metric_function((logits, labels))

    assert "accuracy" in result
    assert "classification_report" not in result


def test_generate_explanation_returns_text():
    result = bert_baseline.generate_explanation(
        "email text",
        0,
        FakeExplanationModel(),
        FakeTokenizer(),
    )

    assert result == "simple explanation"


def test_create_explanations_makes_list(monkeypatch):
    monkeypatch.setattr(
        bert_baseline,
        "generate_explanation",
        lambda text, label, exp_model, exp_tokenizer: f"why {label}",
    )

    result = bert_baseline.create_explanations(
        ["one", "two"],
        [0, 1],
        FakeExplanationModel(),
        FakeTokenizer(),
    )

    assert result == ["why 0", "why 1"]
