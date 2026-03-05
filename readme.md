# 📈 Stock Price Prediction API

A full-stack ML application that predicts next-day stock prices using multiple machine learning models, with a FastAPI backend and interactive Streamlit dashboard.

## 🚀 Live Demo
**Dashboard:** https://stock-price-predictor-1.streamlit.app

## 🚀 Features
- Real-time stock data fetching via Yahoo Finance
- 30+ technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- 4 ML models: Linear Regression, Ridge, Random Forest, Gradient Boosting
- Auto-selects best model based on R² score
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

## 📊 Sample Results (AAPL)
- Linear Regression R²: **0.9836**
- Ridge Regression R²: **0.9829**
- Prediction accuracy within ~$3.65 RMSE on test set
