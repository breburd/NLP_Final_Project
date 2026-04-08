import os
from pathlib import Path
from typing import List, Dict


class EnronDatasetLoader:
    """
    Loads and processes emails from the Enron dataset.
    
    Download the Enron dataset from Git LFS and uses this class to load
    the dataset and all the emails for each user.
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def _read_email(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def load_emails(self) -> List[Dict]:
        """
        Recursively finds emails within the Enron dataset file structure:
        maildir/<user>/<subdirs>/*
        """
        emails = []

        for user_dir in self.root_dir.iterdir():
            if not user_dir.is_dir():
                continue

            user_name = user_dir.name

            for subdir, _, files in os.walk(user_dir):
                for file in files:
                    file_path = Path(subdir) / file
                    raw_email = self._read_email(file_path)

                    if raw_email.strip():
                        emails.append({
                            "user": user_name,
                            "path": str(file_path),
                            "raw": raw_email
                        })

        return emails
    