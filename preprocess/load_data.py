import os
from pathlib import Path
from typing import List, Dict


class EnronDatasetLoader:
    """
    Load and process raw emails from the Enron dataset directory.
    Traverses the Enron maildir structure and extracts raw email
    content for each user. 

    Args:
        root_dir (str): Path to the root Enron maildir directory.

    Attributes:
        root_dir (pathlib.Path): Root directory as a Path object.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def _read_email(self, file_path: Path) -> str:
        """
        Read the contents of a single email file.

        Attempts to open and read the file using 'latin-1' encoding.
        If an error occurs, an empty string is returned.

        Args:
            file_path (pathlib.Path): Path to the email file.

        Returns:
            str: Raw email content, or an empty string if reading fails.
        """

        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def load_emails(self) -> List[Dict]:
        """
        Load all emails from the dataset directory.

        Recursively traverses user directories and collects raw email
        contents along with metadata such as user and file path.

        Returns:
            list[dict]: List of email records, where each record contains:
                - user (str): User identifier (folder name).
                - path (str): File path to the email.
                - raw (str): Raw email content.
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
    