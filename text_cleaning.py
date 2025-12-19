"""
text_cleaning.py

Utility function for cleaning raw news text before sentiment analysis.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean raw text for sentiment analysis.

    Steps
    -----
    1) Remove HTML tags.
    2) Remove URLs.
    3) Remove non-alphanumeric characters, keep basic punctuation.
    4) Lowercase all characters.
    5) Collapse multiple spaces.
    """
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove non-alphanumeric characters (keep basic punctuation and spaces)
    text = re.sub(r"[^0-9a-zA-Z\s\.\,\-\+\%\$]", " ", text)

    # Lowercase
    text = text.lower()

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    # Simple self-test
    raw = "Dow jumps 300 points on FED news! <b>Click here</b>: https://example.com"
    print("RAW   :", raw)
    print("CLEAN :", clean_text(raw))
