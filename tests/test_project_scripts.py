from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import run_baselines
from conftest import load_module_from_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fix_filenames_renames_file_with_dot(monkeypatch, temp_workspace):
    monkeypatch.setattr("os.walk", lambda root: [])
    rename_files = load_module_from_path(
        "rename_files_for_test",
        PROJECT_ROOT / "rename_files.py",
    )

    old_name = "email.txt."
    new_name = "email.txt"
    fake_walk = [(str(temp_workspace), [], [old_name])]
    rename_calls = []

    monkeypatch.setattr(rename_files.os, "walk", lambda root: fake_walk)
    monkeypatch.setattr(rename_files.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        rename_files.os,
        "rename",
        lambda old_path, new_path: rename_calls.append((old_path, new_path)),
    )

    rename_files.fix_filenames(str(temp_workspace))

    assert len(rename_calls) == 1
    assert new_name in rename_calls[0][1]


def test_print_dataset_info_prints_numbers(capsys, monkeypatch, temp_workspace):
    fake_df = pd.DataFrame(
        {
            "from": ["a", "b"],
            "to": ["c", "d"],
            "label": [1, 0],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda path: fake_df)
    dataset_info = load_module_from_path(
        "dataset_info_for_test",
        PROJECT_ROOT / "preprocess" / "dataset_info.py",
    )

    csv_path = temp_workspace / "sample.csv"
    fake_df.to_csv(csv_path, index=False)
    dataset_info.print_dataset_info(str(csv_path))

    captured = capsys.readouterr()

    assert "Number of emails: 2" in captured.out
    assert "Number of privileged emails: 1" in captured.out


def test_run_one_command_success(monkeypatch):
    monkeypatch.setattr(
        run_baselines.subprocess,
        "run",
        lambda command_list, cwd=None: SimpleNamespace(returncode=0),
    )

    run_baselines.run_one_command(["python", "demo.py"])


def test_run_one_command_failure(monkeypatch):
    monkeypatch.setattr(
        run_baselines.subprocess,
        "run",
        lambda command_list, cwd=None: SimpleNamespace(returncode=5),
    )

    with pytest.raises(SystemExit):
        run_baselines.run_one_command(["python", "demo.py"])


def test_run_baselines_main_runs_all_commands(monkeypatch):
    args = Namespace(
        dataset_path="fake_dataset",
        data_path="fake_data.csv",
        run_preprocess=False,
        run_keyword=False,
        run_logistic=False,
        run_bert=False,
        run_all=True,
        bert_train_size=10,
        bert_valid_size=5,
        bert_test_size=5,
    )
    commands = []

    monkeypatch.setattr(
        run_baselines.argparse.ArgumentParser,
        "parse_args",
        lambda self: args,
    )
    monkeypatch.setattr(run_baselines, "run_one_command", lambda cmd: commands.append(cmd))

    run_baselines.main()

    assert len(commands) == 4
    assert "preprocess/preprocess.py" in commands[0]
    assert "models/keyword_filter.py" in commands[1]
    assert "models/logistic_regression.py" in commands[2]
    assert "models/bert_baseline.py" in commands[3]


def test_run_baselines_main_stops_when_dataset_path_is_missing(monkeypatch):
    args = Namespace(
        dataset_path="",
        data_path="fake_data.csv",
        run_preprocess=True,
        run_keyword=False,
        run_logistic=False,
        run_bert=False,
        run_all=False,
        bert_train_size=10,
        bert_valid_size=5,
        bert_test_size=5,
    )

    monkeypatch.setattr(
        run_baselines.argparse.ArgumentParser,
        "parse_args",
        lambda self: args,
    )

    with pytest.raises(SystemExit):
        run_baselines.main()
