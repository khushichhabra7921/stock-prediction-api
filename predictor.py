import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StockPredictor:
    def __init__(self, symbol='AAPL', period='5y'):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None   # rows with a known next-day target
        self.latest = None     # today's row: features known, target unknown
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
        """Model features. Price-level indicators are divided by Close so they are scale-free."""
        df = self.data.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean() / df['Close']
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean() / df['Close']

        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()

        df['RSI'] = self._rsi(df['Close'])

        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = (exp1 - exp2) / df['Close']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        bb_middle = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df['BB_Width'] = (bb_upper - bb_lower) / df['Close']
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

        df['Price_Position_20'] = df['Close'] / df['Close'].rolling(window=20).max()
        df['Price_Position_50'] = df['Close'] / df['Close'].rolling(window=50).max()

        volume_sma_10 = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / volume_sma_10

        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag) / df['Close']
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag) / volume_sma_10
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)

        # Target: next day's return
        df['Target'] = df['Close'].pct_change().shift(-1)
        return df

    @staticmethod
    def _rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        return 100 - (100 / (1 + gain / loss))

    def prepare_features(self):
        df = self.calculate_technical_indicators()
        feature_cols = [col for col in df.columns if col not in
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Dividends', 'Stock Splits',
                        'Capital Gains']]
        df = df[feature_cols + ['Target', 'Close']].dropna(subset=feature_cols)
        self.latest = df.iloc[[-1]]                   # today: features known, target unknown
        self.features = df.dropna(subset=['Target'])  # rows with a known next-day target
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

        y_true = y_test.values
        baseline_rmse = np.sqrt(np.mean(y_true ** 2))  # "price won't change" baseline

        for name, (model, use_scaled) in models.items():
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            direction_acc = np.mean(np.sign(y_pred) == np.sign(y_true))

            self.models[name] = (model, use_scaled)
            self.results[name] = {
                'Direction_Acc': round(float(direction_acc), 4),
                'RMSE': round(float(rmse), 5),
                'Baseline_RMSE': round(float(baseline_rmse), 5),
                'MAE': round(float(mae), 5),
                'R2': round(float(r2), 4),
            }

        self.test_size = len(y_test)
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['Direction_Acc'])
        return self.results

    def predict_next_day(self):
        if not self.models:
            self.train()

        latest = self.latest[self.feature_cols].values
        current_price = float(self.latest['Close'].iloc[0])

        predictions = {}
        predicted_returns = {}
        for name, (model, use_scaled) in self.models.items():
            X = self.scalers['standard'].transform(latest) if use_scaled else latest
            ret = float(model.predict(X)[0])
            predicted_returns[name] = round(ret * 100, 3)
            predictions[name] = round(current_price * (1 + ret), 2)

        best_pred = predictions[self.best_model_name]
        change = round(best_pred - current_price, 2)

        return {
            "symbol": self.symbol,
            "as_of": str(self.latest.index[0].date()),
            "current_price": round(current_price, 2),
            "predicted_next_day": best_pred,
            "predicted_return_percent": predicted_returns[self.best_model_name],
            "best_model": self.best_model_name,
            "change": change,
            "change_percent": predicted_returns[self.best_model_name],
            "test_days": self.test_size,
            "all_model_predictions": predictions,
            "all_model_returns_percent": predicted_returns,
            "model_accuracies": self.results
        }

    def get_indicators(self):
        """Latest indicator values in dollar terms, for display."""
        if self.data is None:
            self.fetch_data()
        close = self.data['Close']
        sma_20 = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = sma_20 + bb_std * 2
        bb_lower = sma_20 - bb_std * 2
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        volume_ratio = self.data['Volume'] / self.data['Volume'].rolling(window=10).mean()
        price = float(close.iloc[-1])
        return {
            "symbol": self.symbol,
            "current_price": round(price, 2),
            "RSI": round(float(self._rsi(close).iloc[-1]), 2),
            "MACD": round(float(macd.iloc[-1]), 4),
            "MACD_Signal": round(float(macd.ewm(span=9).mean().iloc[-1]), 4),
            "BB_Upper": round(float(bb_upper.iloc[-1]), 2),
            "BB_Lower": round(float(bb_lower.iloc[-1]), 2),
            "BB_Position": round(float((price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])), 4),
            "SMA_20": round(float(sma_20.iloc[-1]), 2),
            "SMA_50": round(float(close.rolling(window=50).mean().iloc[-1]), 2),
            "EMA_20": round(float(close.ewm(span=20).mean().iloc[-1]), 2),
            "Volatility_10": round(float(close.pct_change().rolling(window=10).std().iloc[-1]), 6),
            "Volume_Ratio": round(float(volume_ratio.iloc[-1]), 4),
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
