"""
macro_agent.py

Macro & Fundamentals Agent for fixed-income portfolio positioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import os

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import shap

try:
    from fredapi import Fred
except ImportError:
    Fred = None
    print("Warning: fredapi not installed. Install with: pip install fredapi")


@dataclass
class MacroIndicator:
    name: str
    fred_code: str
    weight: float


class MacroAgent:
    FRED_SERIES = {
        "CPI": "CPIAUCSL",
        "GDP": "GDPC1",
        "UNEMPLOYMENT": "UNRATE",
        "ISM_PMI": "NAPM",
        "TREASURY_10Y": "DGS10",
        "TREASURY_2Y": "DGS2",
        "BAA": "BAMLC0A0CM",
        "AAA": "AAA",
    }

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        lag_periods: int = 1,
        vol_window: int = 12,
        random_state: int = 42,
    ):
        self.lag_periods = lag_periods
        self.vol_window = vol_window
        self.random_state = random_state

        if fred_api_key is None:
            fred_api_key = os.getenv("FRED_API_KEY")
        
        if fred_api_key is None:
            raise ValueError(
                "FRED API key required. Provide via fred_api_key parameter "
                "or set FRED_API_KEY environment variable."
            )

        if Fred is None:
            raise ImportError(
                "fredapi package required. Install with: pip install fredapi"
            )

        self.fred = Fred(api_key=fred_api_key)

        if weights is None:
            self.weights = {
                "CPI": 0.15,
                "GDP": 0.15,
                "UNEMPLOYMENT": 0.15,
                "ISM_PMI": 0.15,
                "YIELD_SPREAD": 0.20,
                "CREDIT_SPREAD": 0.20,
            }
        else:
            self.weights = weights

        self.clf = None
        self.shap_explainer = None
        self.feature_names = None

    def run(
        self,
        start_date: str,
        end_date: str,
        output_path: Optional[str] = "macro_signals.csv",
    ) -> pd.DataFrame:
        raw_data = self._fetch_fred_data(start_date, end_date)
        aligned_data = self._align_timeline(raw_data)
        features = self._engineer_features(aligned_data)
        signals = self._classify_and_generate_signals(features)

        if output_path:
            signals.to_csv(output_path, index=True)
            print(f"Macro signals saved to {output_path}")

        return signals

    def _fetch_fred_data(
        self, start_date: str, end_date: str
    ) -> Dict[str, pd.Series]:
        data = {}
        print("Fetching data from FRED API...")

        for name, series_id in self.FRED_SERIES.items():
            try:
                series = self.fred.get_series(
                    series_id, start=start_date, end=end_date
                )
                if series is not None and len(series) > 0:
                    data[name] = series
                    print(f"  ✓ {name} ({series_id}): {len(series)} observations")
                else:
                    print(f"  ✗ {name} ({series_id}): No data")
            except Exception as e:
                print(f"  ✗ {name} ({series_id}): Error - {e}")

        if "TREASURY_10Y" in data and "TREASURY_2Y" in data:
            data["YIELD_SPREAD"] = data["TREASURY_10Y"] - data["TREASURY_2Y"]
            print("  ✓ YIELD_SPREAD computed")

        if "BAA" in data and "AAA" in data:
            data["CREDIT_SPREAD"] = data["BAA"] - data["AAA"]
            print("  ✓ CREDIT_SPREAD computed")

        if "GDP" in data:
            gdp = data["GDP"]
            data["GDP_GROWTH"] = gdp.pct_change(periods=4) * 100
            print("  ✓ GDP_GROWTH computed")

        return data

    def _align_timeline(
        self, raw_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        if not raw_data:
            raise ValueError("No data fetched from FRED API.")

        start = min(s.index.min() for s in raw_data.values())
        end = max(s.index.max() for s in raw_data.values())
        monthly_index = pd.date_range(
            start=start, end=end, freq="MS", tz="UTC"
        )

        aligned = pd.DataFrame(index=monthly_index)
        for name, series in raw_data.items():
            series_monthly = series.resample("MS").last()
            aligned[name] = series_monthly.reindex(monthly_index).interpolate(
                method="time"
            )

        aligned = aligned.dropna(thresh=len(aligned.columns) * 0.5)
        print(f"Aligned data: {len(aligned)} monthly observations")
        return aligned

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = data.copy()

        z_scores = {}
        for col in data.columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0:
                z_scores[f"{col}_z"] = (data[col] - mean_val) / std_val
            else:
                z_scores[f"{col}_z"] = 0.0

        z_df = pd.DataFrame(z_scores, index=data.index)

        M_score = pd.Series(0.0, index=data.index)
        for indicator, weight in self.weights.items():
            if indicator == "YIELD_SPREAD":
                z_col = "YIELD_SPREAD_z"
            elif indicator == "CREDIT_SPREAD":
                z_col = "CREDIT_SPREAD_z"
            elif indicator == "GDP":
                z_col = "GDP_GROWTH_z" if "GDP_GROWTH_z" in z_df.columns else "GDP_z"
            else:
                z_col = f"{indicator}_z"

            if z_col in z_df.columns:
                M_score += weight * z_df[z_col]

        features["M_score"] = M_score

        for lag in range(1, self.lag_periods + 1):
            features[f"M_score_lag{lag}"] = M_score.shift(lag)

        features["M_score_vol"] = M_score.rolling(
            window=self.vol_window, min_periods=1
        ).std()

        feature_df = pd.concat([z_df, features[["M_score"]]], axis=1)
        if self.lag_periods > 0:
            lag_cols = [f"M_score_lag{i}" for i in range(1, self.lag_periods + 1)]
            feature_df = pd.concat([feature_df, features[lag_cols]], axis=1)
        feature_df["M_score_vol"] = features["M_score_vol"]

        self.feature_names = feature_df.columns.tolist()
        return feature_df

    def _classify_and_generate_signals(
        self, features: pd.DataFrame
    ) -> pd.DataFrame:
        X_cols = ["M_score", "M_score_vol"]
        if self.lag_periods > 0:
            X_cols.extend(
                [f"M_score_lag{i}" for i in range(1, self.lag_periods + 1)]
            )

        valid_idx = features[X_cols].dropna().index
        if len(valid_idx) < 10:
            raise ValueError("Insufficient data for classification.")

        X = features.loc[valid_idx, X_cols].values
        y = self._generate_synthetic_labels(features.loc[valid_idx, "M_score"])

        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X[:1], y[:1]

        self.clf = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
        )
        self.clf.fit(X_train, y_train)

        predictions = self.clf.predict(X)
        probabilities = self.clf.predict_proba(X)

        regime_map = {0: "long_duration", 1: "short_duration", 2: "credit_exposure"}
        regimes = [regime_map[pred] for pred in predictions]
        confidences = np.max(probabilities, axis=1)

        signals = pd.DataFrame(
            index=valid_idx,
            data={
                "regime": regimes,
                "confidence": confidences,
                "M_score": features.loc[valid_idx, "M_score"].values,
                "p_long": probabilities[:, 0],
                "p_short": probabilities[:, 1],
                "p_credit": probabilities[:, 2],
            },
        )

        for col in features.columns:
            if col.endswith("_z") or col in ["M_score", "M_score_vol"]:
                if col not in signals.columns:
                    signals[col] = features.loc[valid_idx, col].values

        try:
            self.shap_explainer = shap.TreeExplainer(self.clf)
            shap_values = self.shap_explainer.shap_values(X)
            signals["shap_sum"] = np.sum(np.abs(shap_values[0]), axis=1)
        except Exception as e:
            print(f"Warning: SHAP initialization failed - {e}")

        return signals.sort_index()

    def _generate_synthetic_labels(self, M_scores: pd.Series) -> np.ndarray:
        labels = []
        for score in M_scores:
            if score > 0.5:
                labels.append(0)
            elif score < -0.5:
                labels.append(1)
            else:
                labels.append(2)
        return np.array(labels)

    def get_shap_values(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        if self.shap_explainer is None or self.clf is None:
            return None

        X_cols = ["M_score", "M_score_vol"]
        if self.lag_periods > 0:
            X_cols.extend(
                [f"M_score_lag{i}" for i in range(1, self.lag_periods + 1)]
            )

        X = features[X_cols].dropna().values
        shap_values = self.shap_explainer.shap_values(X)
        return shap_values


if __name__ == "__main__":
    import os

    api_key = os.getenv("FRED_API_KEY")
    if api_key is None:
        print(
            "Warning: FRED_API_KEY not set. "
            "Set it as an environment variable or pass fred_api_key parameter."
        )
        print("Example usage:")
        print("  export FRED_API_KEY='your_key_here'")
        print("  python macro_agent.py")
    else:
        agent = MacroAgent(
            fred_api_key=api_key,
            lag_periods=1,
            vol_window=12,
        )

        signals = agent.run(
            start_date="2020-01-01",
            end_date="2024-12-31",
            output_path="macro_signals.csv",
        )

        print("\nMacro Signals Summary:")
        print(signals.head(10))
        print(f"\nRegime distribution:")
        print(signals["regime"].value_counts())
        print(f"\nAverage confidence: {signals['confidence'].mean():.3f}")

