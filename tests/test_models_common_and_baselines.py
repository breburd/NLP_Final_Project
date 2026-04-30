from argparse import Namespace
import numpy as np
import pandas as pd
import pytest
import torch

import common
import keyword_filter
import logistic_regression


def test_seed_everything_makes_repeatable_numbers():
    common.seed_everything(7)
    first_numbers = (
        np.random.rand(),
        torch.rand(1).item(),
    )

    common.seed_everything(7)
    second_numbers = (
        np.random.rand(),
        torch.rand(1).item(),
    )

    assert first_numbers == second_numbers


def test_print_gpu_memory_prints_when_gpu_is_available(monkeypatch, capsys):
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(common.torch.cuda, "memory_allocated", lambda index: 1)
    monkeypatch.setattr(common.torch.cuda, "memory_reserved", lambda index: 2)
    monkeypatch.setattr(common.torch.cuda, "max_memory_reserved", lambda index: 3)
    monkeypatch.setattr(common.subprocess, "check_output", lambda command: b"nvidia output")

    common.print_gpu_memory()
    captured = capsys.readouterr()

    assert "torch.cuda.memory_allocated" in captured.out
    assert "nvidia output" in captured.out


def test_make_folder_creates_directory(temp_workspace):
    folder_path = temp_workspace / "my_output"
    result = common.make_folder(folder_path)

    assert folder_path.exists()
    assert result == folder_path


def test_fix_csv_limit_retries_after_overflow(monkeypatch):
    import csv

    calls = []

    def fake_field_size_limit(size):
        calls.append(size)
        if len(calls) == 1:
            raise OverflowError("too big")
        return None

    monkeypatch.setattr(csv, "field_size_limit", fake_field_size_limit)

    common.fix_csv_limit()

    assert len(calls) >= 2


def test_load_data_combines_subject_and_body(temp_workspace):
    csv_path = temp_workspace / "emails.csv"
    df = pd.DataFrame(
        {
            "from": ["amy"],
            "to": ["bob"],
            "subject": ["Hello"],
            "body": ["World"],
            "label": [1],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded_df = common.load_data(csv_path)

    assert loaded_df.loc[0, "text"] == "Hello\nWorld"
    assert loaded_df.loc[0, "label"] == 1


def test_split_data_returns_three_parts():
    df = pd.DataFrame(
        {
            "from": [f"from_{i}" for i in range(30)],
            "to": [f"to_{i}" for i in range(30)],
            "subject": ["s"] * 30,
            "body": ["b"] * 30,
            "label": [0, 1] * 15,
            "text": [f"row {i}" for i in range(30)],
        }
    )

    train_df, valid_df, test_df = common.split_data(
        df,
        test_size=0.2,
        valid_size=0.1,
        seed=42,
    )

    assert len(train_df) + len(valid_df) + len(test_df) == len(df)
    assert set(train_df["pair_id"]).isdisjoint(set(valid_df["pair_id"]))
    assert set(train_df["pair_id"]).isdisjoint(set(test_df["pair_id"]))


def test_get_scores_returns_metric_dictionary():
    scores = common.get_scores([1, 0, 1], [1, 0, 0])

    assert "accuracy" in scores
    assert "classification_report" in scores
    assert isinstance(scores["f1"], float)


def test_save_json_writes_file(temp_workspace):
    output_path = temp_workspace / "results" / "data.json"

    common.save_json({"name": "test"}, output_path)

    assert output_path.exists()
    assert '"name": "test"' in output_path.read_text(encoding="utf-8")


def test_run_keyword_model_finds_keywords():
    text_series = pd.Series(
        [
            "This message is confidential and legal advice.",
            "Normal team lunch email.",
        ]
    )

    result = keyword_filter.run_keyword_model(text_series)

    assert result.tolist() == [1, 0]


def test_keyword_filter_main_saves_results(monkeypatch, temp_workspace):
    args = Namespace(
        data_path="fake.csv",
        output_dir=str(temp_workspace),
        seed=42,
    )
    fake_df = pd.DataFrame(
        {
            "text": ["privileged note", "hello there"],
            "label": [1, 0],
            "subject": ["one", "two"],
        }
    )
    saved = {}

    monkeypatch.setattr(
        keyword_filter.argparse.ArgumentParser,
        "parse_args",
        lambda self: args,
    )
    monkeypatch.setattr(keyword_filter, "load_data", lambda path: fake_df)
    monkeypatch.setattr(
        keyword_filter,
        "split_data",
        lambda df, seed=None: (fake_df, fake_df, fake_df),
    )
    monkeypatch.setattr(
        keyword_filter,
        "save_json",
        lambda data, path: saved.update({"data": data, "path": path}),
    )

    keyword_filter.main()

    assert saved["data"]["baseline"] == "keyword_filter"
    assert (temp_workspace / "sample_predictions.json").exists()


def test_build_model_has_two_steps():
    model = logistic_regression.build_model(100)

    assert list(model.named_steps.keys()) == ["tfidf", "clf"]


def test_logistic_main_saves_metrics(monkeypatch, temp_workspace):
    args = Namespace(
        data_path="fake.csv",
        output_dir=str(temp_workspace),
        max_features=500,
        seed=42,
    )
    fake_df = pd.DataFrame(
        {
            "text": ["email one", "email two"],
            "label": [0, 1],
        }
    )
    saved = {}

    class FakeModel:
        def fit(self, texts, labels):
            self.was_fit = True

        def predict(self, texts):
            return np.array([0, 1])

    monkeypatch.setattr(
        logistic_regression.argparse.ArgumentParser,
        "parse_args",
        lambda self: args,
    )
    monkeypatch.setattr(logistic_regression, "load_data", lambda path: fake_df)
    monkeypatch.setattr(
        logistic_regression,
        "split_data",
        lambda df, seed=None: (fake_df, fake_df, fake_df),
    )
    monkeypatch.setattr(logistic_regression, "build_model", lambda value: FakeModel())
    monkeypatch.setattr(
        logistic_regression,
        "save_json",
        lambda data, path: saved.update({"data": data, "path": path}),
    )
    monkeypatch.setattr(logistic_regression.pickle, "dump", lambda model, file_obj: None)

    logistic_regression.main()

    assert saved["data"]["baseline"] == "logistic_regression"
    assert (temp_workspace / "model.pkl").exists()
