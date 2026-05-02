"""
Weak supervision pipeline for labeling Enron emails.

Uses Snorkel labeling functions (LFs) to generate noisy labels based
on heuristic rules (keywords, disclaimers, participants, and length).
A LabelModel is trained to combine LF outputs into probabilistic labels.
"""

from snorkel.labeling import labeling_function
import pandas as pd
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from load_data import EnronDatasetLoader
from parser import EmailParser

ABSTAIN = -1
NOT_PRIV = 0
PRIV = 1


LEGAL_KEYWORDS = [
    "attorney-client",
    "privileged",
    "confidential",
    "legal advice",
    "counsel",  ## TODO -> based on other research we may need to remove this as it may be too broad
    "litigation"
]


@labeling_function()
def lf_legal_keywords(x):
    """
    Label emails as privileged based on legal keyword matches.

    Checks whether predefined legal-related keywords appear in the
    subject or body of the email.

    Args:
        x (pandas.Series): Row containing email fields.

    Returns:
        int: PRIV if a keyword is found, otherwise ABSTAIN.
    """
        
    text = (x["subject"] + " " + x["body"]).lower()
    return PRIV if any(k in text for k in LEGAL_KEYWORDS) else ABSTAIN


@labeling_function()
def lf_disclaimer(x):
    """
    Label emails as privileged based on disclaimer phrases.

    Detects common legal disclaimer language that often appears in
    privileged communications.

    Args:
        x (pandas.Series): Row containing email fields.

    Returns:
        int: PRIV if disclaimer text is detected, otherwise ABSTAIN.
    """

    text = x["body"].lower()
    return PRIV if "may contain privileged" in text else ABSTAIN


@labeling_function()
def lf_lawyer_email(x):
    """
    Label emails as privileged based on participant email addresses.

    Flags emails where sender or recipient addresses contain legal-
    related terms ('law', 'legal').

    Args:
        x (pandas.Series): Row containing email fields.

    Returns:
        int: PRIV if legal-related terms are found, otherwise ABSTAIN.

    Notes:
        This is a high-recall but noisy heuristic.
    """

    participants = (x["from"] + " " + x["to"]).lower()
    return PRIV if "law" in participants or "legal" in participants else ABSTAIN


@labeling_function()
def lf_short_email(x):
    """
    Label emails as not privileged based on length.

    Assumes very short emails are unlikely to contain detailed legal
    discussions and labels them as non-privileged.

    Args:
        x (pandas.Series): Row containing email fields.

    Returns:
        int: NOT_PRIV if email body is short, otherwise ABSTAIN.
    """
    
    return NOT_PRIV if len(x["body"]) < 50 else ABSTAIN

if __name__ == "__main__":
    # Use Command Line Argument for dataset path
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Enron emails and apply weak labeling.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Enron dataset root directory")
    args = parser.parse_args()
    # Load dataset
    loader = EnronDatasetLoader(args.dataset_path)
    raw_emails = loader.load_emails()

    # Parse emails
    parser = EmailParser()
    parsed = [parser.parse(e["raw"]) for e in raw_emails]

    df = pd.DataFrame(parsed)

    # Apply labeling functions
    lfs = [
        lf_legal_keywords,
        lf_disclaimer,
        lf_lawyer_email,
        lf_short_email
    ]

    applier = PandasLFApplier(lfs=lfs)
    L = applier.apply(df)

    # Train label model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L, n_epochs=500, log_freq=100)

    # Get probabilistic labels
    df["prob_label"] = label_model.predict_proba(L)[:, 1]

    # Convert to hard labels if needed
    df["label"] = (df["prob_label"] > 0.5).astype(int)

    # save the DataFrame with labels for downstream use
    df.to_csv("enron_emails_labeled.csv", index=False)
