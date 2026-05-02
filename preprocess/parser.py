import re
from typing import Dict


class EmailParser:
    """
    Parse raw email text into structured fields.

    Extracts common email components such as sender, recipient,
    subject, and body from raw Enron email text. Includes basic
    text cleaning for downstream processing.
    """

    def clean(self, text: str) -> str:
        """
        Clean raw email body text.

        Removes excessive whitespace and normalizes spacing to improve
        text consistency for modeling and analysis.

        Args:
            text (str): Raw email body text.

        Returns:
            str: Cleaned text with normalized whitespace.
        """

        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def parse(self, raw_email: str) -> Dict:
        """
        Extract structured fields from raw email text.

        Parses common header fields (From, To, Subject) and separates
        the email body. Applies basic cleaning to the body text.

        Args:
            raw_email (str): Full raw email content.

        Returns:
            dict: Dictionary containing:
                - from (str): Sender of the email.
                - to (str): Recipient of the email.
                - subject (str): Email subject line.
                - body (str): Cleaned email body text.
        """

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
    