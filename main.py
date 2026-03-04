from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predictor import StockPredictor

app = FastAPI(
    title="Stock Price Prediction API",
    description="ML-powered stock price prediction with technical indicators",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Stock Price Prediction API", "docs": "/docs"}

@app.get("/predict/{symbol}")
def predict(symbol: str, period: str = "5y"):
    """Train models and predict next day closing price for a stock"""
    try:
        predictor = StockPredictor(symbol=symbol, period=period)
        result = predictor.predict_next_day()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/indicators/{symbol}")
def indicators(symbol: str):
    """Get latest technical indicators for a stock"""
    try:
        predictor = StockPredictor(symbol=symbol)
        return predictor.get_indicators()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

@app.get("/history/{symbol}")
def history(symbol: str, days: int = 90):
    """Get price history for a stock"""
    try:
        predictor = StockPredictor(symbol=symbol)
        data = predictor.get_price_history(days=days)
        return {"symbol": symbol.upper(), "days": days, "history": data}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")