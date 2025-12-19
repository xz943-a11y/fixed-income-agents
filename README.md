# Fixed-Income Portfolio Agents

Multi-agent system for fixed-income portfolio positioning and risk management.

## Agents

### 1. News Sentiment Agent (`news_sentiment_agent.py`)
FinBERT-based news sentiment analysis agent that processes financial news and generates sentiment signals.

**Features:**
- Text cleaning and preprocessing
- FinBERT sentiment classification
- Daily sentiment aggregation
- Rolling moving average and volatility calculation

**Usage:**
```python
from news_sentiment_agent import NewsSentimentAgent

agent = NewsSentimentAgent(window=5, device=-1)
signals = agent.run(df)  # df with 'timestamp' and 'text' columns
```

### 2. Macro & Fundamentals Agent (`macro_agent.py`)
Macroeconomic analysis agent that generates structural signals for fixed-income portfolio positioning.

**Features:**
- FRED API data acquisition (CPI, GDP, Unemployment, ISM PMI, etc.)
- Feature engineering (z-score normalization, lags, volatilities)
- Decision Tree classification for market regime prediction
- SHAP values for interpretability

**Usage:**
```python
from macro_agent import MacroAgent
import os

agent = MacroAgent(
    fred_api_key=os.getenv("FRED_API_KEY"),
    lag_periods=1,
    vol_window=12,
)

signals = agent.run(
    start_date="2020-01-01",
    end_date="2024-12-31",
    output_path="macro_signals.csv",
)
```

## Requirements

```bash
pip install pandas numpy scikit-learn shap transformers fredapi
```

## Setup

1. Set FRED API key (for Macro Agent):
```bash
export FRED_API_KEY='your_fred_api_key_here'
```

2. Get FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html

## License

MIT

