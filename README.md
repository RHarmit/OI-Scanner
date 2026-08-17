# NSE OI + Chaikin Money Flow Scanner

Streamlit scanner for NSE individual stock-futures underlyings combining 3-day open-interest expansion with Chaikin Money Flow confirmation.

## Core signal

- Aggregated stock-futures open interest across expiries
- 3-trading-day OI change and daily persistence
- 20-session Chaikin Money Flow
- Price/OI buildup classification
- Cash-market liquidity and volume filters
- 0-100 ranking score

Open interest applies to NSE F&O stocks, not cash-only NSE stocks.

## Deploy on Streamlit Community Cloud

Use:

- Repository: `RHarmit/OI-Scanner`
- Branch: `main`
- Main file: `app.py`

No API key or secrets are required. The app reads NSE UDiFF EOD bhavcopy data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The score is a ranking heuristic, not a probability of profit or investment recommendation.
