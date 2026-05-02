"""
Unit tests for BERT baseline components.

Includes tests for dataset handling, training utilities, evaluation
metrics, and explanation generation. Uses mock objects to isolate
functionality and avoid dependency on external models.
"""

from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import bert_baseline


class FakeTokenizer:
    """
    Mock tokenizer for testing.

    Simulates minimal behavior of a Hugging Face tokenizer, including
    encoding inputs and decoding outputs, without requiring the actual
    transformers library.
    """

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
        """
        Simulate tokenization with fixed outputs.

        Args:
            text (str): Input text to tokenize.

        Returns:
            dict: Dictionary containing mock input_ids and attention_mask tensors.
        """

        return {
            "input_ids": torch.tensor([[1, 2, 3, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        }

    def __call__(self, prompt, return_tensors="pt", truncation=True):
        """
        Simulate tokenizer call interface.

        Args:
            prompt (str): Input text prompt.

        Returns:
            dict: Dictionary containing mock tokenized tensors.
        """

        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, values, skip_special_tokens=True):
        """
        Simulate decoding token IDs into text.

        Args:
            values (list or tensor): Token IDs.
            skip_special_tokens (bool, optional): Ignored.

        Returns:
            str: Fixed mock explanation string.
        """

        return "simple explanation"


class FakeExplanationModel:
    """
    Mock explanation model for testing.

    Simulates a sequence-to-sequence model used for generating
    explanations without requiring a real model.
    """

    device = "cpu"

    def generate(self, **kwargs):
        """
        Simulate text generation.

        Returns:
            torch.Tensor: Fixed tensor representing generated tokens.
        """
        return torch.tensor([[10, 11, 12]])


def test_enron_dataset_len():
    """
    Verify that EnronDataset returns the correct length.
    """

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
    """
    Verify that dataset __getitem__ returns expected fields and values.
    """

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
    """
    Verify that compute_loss returns a torch.Tensor.

    Uses a fake model to simulate logits output.
    """

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
    """
    Verify that DataFrame is truncated when limit is applied.
    """
        
    df = pd.DataFrame({"value": [1, 2, 3, 4]})

    result = bert_baseline.maybe_take_some_rows(df, 2)

    assert len(result) == 2


def test_metric_function_returns_basic_scores():
    """
    Verify that metric_function returns evaluation metrics without
    classification report.
    """

    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0])

    result = bert_baseline.metric_function((logits, labels))

    assert "accuracy" in result
    assert "classification_report" not in result


def test_generate_explanation_returns_text():
    """
    Verify that generate_explanation returns decoded text output.
    """

    result = bert_baseline.generate_explanation(
        "email text",
        0,
        FakeExplanationModel(),
        FakeTokenizer(),
    )

    assert result == "simple explanation"


def test_create_explanations_makes_list(monkeypatch):
    """
    Verify that create_explanations returns a list of explanations.

    Uses monkeypatch to replace generate_explanation with a mock
    implementation for deterministic behavior.
    """

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
