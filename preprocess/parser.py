import re
from typing import Dict


class EmailParser:
    """
    Basic parser for extracting structured information from raw email text within the Enron dataset.
    """
    def clean(self, text: str) -> str:
        """
        Basic text cleaning: remove extra whitespace, email artifacts, and forwarded content."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def parse(self, raw_email: str) -> Dict:
        """
        Extract structured fields from raw email text."""
        def extract(field):
            match = re.search(rf"{field}:(.*)", raw_email)
            return match.group(1).strip() if match else ""

        body_split = raw_email.split("\n\n", 1)
        body = body_split[1] if len(body_split) > 1 else ""

        return {
            "from": extract("From"),
            "to": extract("To"),
            "subject": extract("Subject"),
            "body": self.clean(body)
        }
    