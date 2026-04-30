import importlib.util
import os
import sys
import types
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESS_DIR = PROJECT_ROOT / "preprocess"


for folder in [PROJECT_ROOT, MODELS_DIR, PREPROCESS_DIR]:
    folder_text = str(folder)
    if folder_text not in sys.path:
        sys.path.insert(0, folder_text)


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_transformers_stub():
    try:
        import transformers 
        return
    except ImportError:
        pass

    transformers = types.ModuleType("transformers")

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class TrainingArguments:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class AutoModelForSeq2SeqLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers.Trainer = Trainer
    transformers.TrainingArguments = TrainingArguments
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForSequenceClassification = (
        AutoModelForSequenceClassification
    )
    transformers.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM
    sys.modules["transformers"] = transformers


def _install_snorkel_stub():
    try:
        import snorkel  # noqa: F401
        return
    except ImportError:
        pass

    snorkel = types.ModuleType("snorkel")
    labeling = types.ModuleType("snorkel.labeling")
    model = types.ModuleType("snorkel.labeling.model")

    def labeling_function():
        def decorator(func):
            return func

        return decorator

    class PandasLFApplier:
        def __init__(self, lfs=None):
            self.lfs = lfs or []

        def apply(self, df):
            return []

    class LabelModel:
        def __init__(self, cardinality=None, verbose=None):
            self.cardinality = cardinality
            self.verbose = verbose

        def fit(self, *args, **kwargs):
            return None

        def predict_proba(self, data):
            return []

    labeling.labeling_function = labeling_function
    labeling.PandasLFApplier = PandasLFApplier
    model.LabelModel = LabelModel

    sys.modules["snorkel"] = snorkel
    sys.modules["snorkel.labeling"] = labeling
    sys.modules["snorkel.labeling.model"] = model


def _install_tqdm_stub():
    try:
        import tqdm  # noqa: F401
        return
    except ImportError:
        pass

    tqdm_module = types.ModuleType("tqdm")

    def tqdm(iterable, total=None):
        return iterable

    tqdm_module.tqdm = tqdm
    sys.modules["tqdm"] = tqdm_module


_install_transformers_stub()
_install_snorkel_stub()
_install_tqdm_stub()


@pytest.fixture
def temp_workspace():
    temp_root = PROJECT_ROOT / "tests_runtime"
    temp_root.mkdir(exist_ok=True)
    temp_dir = temp_root / f"temp_{uuid.uuid4().hex}"
    temp_dir.mkdir()
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            for item in sorted(temp_dir.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    item.rmdir()
            temp_dir.rmdir()
