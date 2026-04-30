from pathlib import Path

from load_data import EnronDatasetLoader
from parser import EmailParser
from conftest import load_module_from_path


PREPROCESS_MODULE = load_module_from_path(
    "preprocess_module_for_tests",
    Path(__file__).resolve().parents[1] / "preprocess" / "preprocess.py",
)


def test_read_email_returns_text(temp_workspace):
    loader = EnronDatasetLoader(str(temp_workspace))
    email_file = temp_workspace / "message.txt"
    email_file.write_text("hello email", encoding="latin-1")

    result = loader._read_email(email_file)

    assert result == "hello email"


def test_read_email_returns_empty_string_for_missing_file(temp_workspace):
    loader = EnronDatasetLoader(str(temp_workspace))

    result = loader._read_email(temp_workspace / "missing.txt")

    assert result == ""


def test_load_emails_collects_files(temp_workspace):
    user_folder = temp_workspace / "person1" / "inbox"
    user_folder.mkdir(parents=True)
    email_file = user_folder / "mail1.txt"
    email_file.write_text("From: a\n\nbody text", encoding="latin-1")

    loader = EnronDatasetLoader(str(temp_workspace))
    emails = loader.load_emails()

    assert len(emails) == 1
    assert emails[0]["user"] == "person1"


def test_clean_removes_extra_spaces():
    parser = EmailParser()

    result = parser.clean("hello   there \n\n friend")

    assert result == "hello there friend"


def test_parse_reads_basic_email_fields():
    parser = EmailParser()
    raw_email = "From: Amy\nTo: Bob\nSubject: Hi\n\nThis is the body."

    result = parser.parse(raw_email)

    assert result["from"] == "Amy"
    assert result["to"] == "Bob"
    assert result["subject"] == "Hi"
    assert result["body"] == "This is the body."


def test_lf_legal_keywords_returns_priv():
    row = {"subject": "Legal Advice", "body": "This is confidential."}

    result = PREPROCESS_MODULE.lf_legal_keywords(row)

    assert result == PREPROCESS_MODULE.PRIV


def test_lf_disclaimer_returns_priv():
    row = {"body": "This may contain privileged information."}

    result = PREPROCESS_MODULE.lf_disclaimer(row)

    assert result == PREPROCESS_MODULE.PRIV


def test_lf_lawyer_email_returns_priv():
    row = {"from": "law.team@company.com", "to": "person@company.com"}

    result = PREPROCESS_MODULE.lf_lawyer_email(row)

    assert result == PREPROCESS_MODULE.PRIV


def test_lf_short_email_returns_not_priv():
    row = {"body": "short note"}

    result = PREPROCESS_MODULE.lf_short_email(row)

    assert result == PREPROCESS_MODULE.NOT_PRIV
