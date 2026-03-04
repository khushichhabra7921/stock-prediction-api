import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StockPredictor:
    def __init__(self, symbol='AAPL', period='5y'):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model_name = None

    def fetch_data(self):
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(period=self.period)
        if self.data.empty:
            raise ValueError(f"No data found for symbol {self.symbol}")
        return True

    def calculate_technical_indicators(self):
        df = self.data.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()

        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))

        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        df['Price_Position_20'] = df['Close'] / df['Close'].rolling(window=20).max()
        df['Price_Position_50'] = df['Close'] / df['Close'].rolling(window=50).max()

        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']

        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)

        df['Target'] = df['Close'].shift(-1)
        return df

    def prepare_features(self):
        df = self.calculate_technical_indicators()
        feature_cols = [col for col in df.columns if col not in
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Dividends', 'Stock Splits']]
        self.features = df[feature_cols + ['Target', 'Close']].dropna()
        return feature_cols

    def train(self):
        self.fetch_data()
        feature_cols = self.prepare_features()

        self.features = self.features.sort_index()
        X = self.features[feature_cols]
        y = self.features['Target']

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        self.feature_cols = feature_cols

        models = {
            'Linear Regression': (LinearRegression(), True),
            'Ridge Regression': (Ridge(alpha=1.0), True),
            'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), False),
            'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
        }

        for name, (model, use_scaled) in models.items():
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.models[name] = (model, use_scaled)
            self.results[name] = {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'R2': round(r2, 4)}

        self.best_model_name = max(self.results, key=lambda x: self.results[x]['R2'])
        return self.results

    def predict_next_day(self):
        if not self.models:
            self.train()

        feature_cols = self.feature_cols
        latest = self.features[feature_cols].iloc[-1].values.reshape(1, -1)

        predictions = {}
        for name, (model, use_scaled) in self.models.items():
            X = self.scalers['standard'].transform(latest) if use_scaled else latest
            pred = model.predict(X)[0]
            predictions[name] = round(float(pred), 2)

        current_price = round(float(self.features['Close'].iloc[-1]), 2)
        best_pred = predictions[self.best_model_name]
        change = round(best_pred - current_price, 2)
        change_pct = round((change / current_price) * 100, 2)

        return {
            "symbol": self.symbol,
            "current_price": current_price,
            "predicted_next_day": best_pred,
            "best_model": self.best_model_name,
            "change": change,
            "change_percent": change_pct,
            "all_model_predictions": predictions,
            "model_accuracies": self.results
        }

    def get_indicators(self):
        if self.data is None:
            self.fetch_data()
        df = self.calculate_technical_indicators()
        latest = df.iloc[-1]
        return {
            "symbol": self.symbol,
            "current_price": round(float(latest['Close']), 2),
            "RSI": round(float(latest['RSI']), 2),
            "MACD": round(float(latest['MACD']), 4),
            "MACD_Signal": round(float(latest['MACD_Signal']), 4),
            "BB_Upper": round(float(latest['BB_Upper']), 2),
            "BB_Lower": round(float(latest['BB_Lower']), 2),
            "BB_Position": round(float(latest['BB_Position']), 4),
            "SMA_20": round(float(latest['SMA_20']), 2),
            "SMA_50": round(float(latest['SMA_50']), 2),
            "EMA_20": round(float(latest['EMA_20']), 2),
            "Volatility_10": round(float(latest['Volatility_10']), 6),
            "Volume_Ratio": round(float(latest['Volume_Ratio']), 4),
        }

    def get_price_history(self, days=90):
        if self.data is None:
            self.fetch_data()
        df = self.data.tail(days)
        return [
            {"date": str(idx.date()), "close": round(float(row['Close']), 2),
             "volume": int(row['Volume'])}
            for idx, row in df.iterrows()
        ]