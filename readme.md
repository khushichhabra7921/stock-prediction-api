# 📈 Stock Price Prediction API

A full-stack ML application that predicts next-day stock returns using multiple machine learning models and evaluates them against a naive baseline, with a FastAPI backend and interactive Streamlit dashboard.

## 🚀 Live Demo
**Dashboard:** https://stock-price-predictor-1.streamlit.app

## 🚀 Features
- Daily stock data fetched from Yahoo Finance
- 30+ technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- 4 ML models: Linear Regression, Ridge, Random Forest, Gradient Boosting
- Predicts next-day **return** (implied price = close × (1 + return)) from today's features
- Reports RMSE against a "price won't change" baseline, plus directional (up/down) accuracy
- Auto-selects best model by directional accuracy on a held-out time-ordered test set
- Interactive Streamlit dashboard with Plotly charts
- REST API with 3 endpoints

## 🛠️ Tech Stack
Python, FastAPI, scikit-learn, Streamlit, Plotly, yfinance, pandas, numpy

## 📡 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/predict/{symbol}` | Train models & predict next-day price |
| GET | `/indicators/{symbol}` | Get latest technical indicators |
| GET | `/history/{symbol}` | Get price history |

## ⚙️ Setup
```bash
pip install -r requirements.txt
```

**Terminal 1 — Start API:**
```bash
uvicorn main:app --reload
```

**Terminal 2 — Start Dashboard:**
```bash
streamlit run frontend.py
```

Open `http://localhost:8501` for the dashboard or `http://localhost:8000/docs` for API docs.

## 📊 Sample Results (AAPL, 5y of data, last 211 trading days held out, as of 2026-09-25)
| Model | Directional accuracy | RMSE (daily return) | No-change baseline RMSE |
|-------|---------------------|---------------------|-------------------------|
| Gradient Boosting | ~53% | 0.0168 | 0.0160 |
| Linear Regression | 52.1% | 0.0167 | 0.0160 |
| Ridge Regression | 52.1% | 0.0161 | 0.0160 |
| Random Forest | ~46–49% | 0.0167 | 0.0160 |

No model beats the no-change baseline on RMSE, and directional accuracy is only slightly above a coin flip (±~3.4 points is one standard error at 211 days). R² on returns is around zero or negative. That is the expected, honest result for next-day stock returns; an earlier version reported R² ≈ 0.98 by predicting price levels, which mostly measures that tomorrow's price is close to today's.
