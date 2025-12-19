"""
news_sentiment_agent.py

FinBERT-based News & Sentiment Agent.

Input  : DataFrame with columns ['timestamp', 'text'] (+ optional 'ticker')
Output : DataFrame indexed by trading_date with columns
         ['S_raw', 'S_MA', 'S_Vol'] corresponding to (S_t, MA_t, Vol_t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

from text_cleaning import clean_text


@dataclass
class NewsItem:
    """
    Single news item.

    Attributes
    ----------
    timestamp : pd.Timestamp
    text : str
    ticker : Optional[str]
    """
    timestamp: pd.Timestamp
    text: str
    ticker: Optional[str] = None


class NewsSentimentAgent:
    """
    FinBERT-based News & Sentiment Agent.

    Steps:
      1) Clean raw text.
      2) Apply FinBERT to get sentiment probabilities.
      3) Compute S_i,t = p_pos - p_neg.
      4) Aggregate to daily S_t.
      5) Compute rolling MA_t and Vol_t.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        window: int = 5,
        device: int = -1,
        min_text_length: int = 5,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name.
        window : int
            Rolling window length W.
        device : int
            Device id for transformers pipeline. -1 for CPU.
        min_text_length : int
            Minimum text length after cleaning.
        """
        self.window = window
        self.min_text_length = min_text_length

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self._clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=device,
        )

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the News & Sentiment pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain:
            - 'timestamp': datetime-like
            - 'text'     : str

        Returns
        -------
        pd.DataFrame
            Index: trading_date
            Columns: 'S_raw', 'S_MA', 'S_Vol'
        """
        df = df.copy()

        if "timestamp" not in df.columns or "text" not in df.columns:
            raise ValueError("Input DataFrame must contain 'timestamp' and 'text' columns.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Text cleaning and filtering
        df["clean_text"] = df["text"].astype(str).apply(clean_text)
        df = df[df["clean_text"].str.len() >= self.min_text_length]
        df = df.drop_duplicates(subset=["clean_text", "timestamp"])

        if df.empty:
            raise ValueError("No valid news items left after cleaning and filtering.")

        # Map to trading date (UTC calendar)
        df["trading_date"] = df["timestamp"].dt.normalize()

        # FinBERT inference
        scores = df["clean_text"].apply(self._score_single_item)
        scores_df = pd.DataFrame(list(scores))
        df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

        # S_i,t = p_pos - p_neg
        df["S_item"] = df["p_pos"] - df["p_neg"]

        # Aggregate to daily S_t
        daily = (
            df.groupby("trading_date")["S_item"]
            .mean()
            .rename("S_raw")
            .to_frame()
            .sort_index()
        )

        # Rolling MA_t and Vol_t
        W = self.window
        daily["S_MA"] = daily["S_raw"].rolling(window=W, min_periods=1).mean()
        daily["S_Vol"] = daily["S_raw"].rolling(window=W, min_periods=2).std(ddof=1)

        return daily

    def _score_single_item(self, text: str) -> dict:
        """
        Run FinBERT on a single cleaned text snippet and
        return probabilities for negative / neutral / positive.
        """
        outputs = self._clf(text)[0]

        prob_map = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for item in outputs:
            label = item["label"].lower()
            score = float(item["score"])
            if "neg" in label:
                prob_map["negative"] = score
            elif "neu" in label:
                prob_map["neutral"] = score
            elif "pos" in label:
                prob_map["positive"] = score

        return {
            "p_neg": prob_map["negative"],
            "p_neu": prob_map["neutral"],
            "p_pos": prob_map["positive"],
        }


if __name__ == "__main__":
    # Minimal usage example
    sample = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-02T14:30:00Z",
                "2024-01-02T15:10:00Z",
                "2024-01-03T09:05:00Z",
            ],
            "text": [
                "Credit spreads tighten as investors turn optimistic on corporate earnings.",
                "High-yield bond market under pressure amid growing default concerns.",
                "Government bond yields fall after dovish comments from the central bank.",
            ],
        }
    )

    agent = NewsSentimentAgent(window=5, device=-1)
    signals = agent.run(sample)
    print(signals)
