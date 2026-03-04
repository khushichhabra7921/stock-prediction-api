import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Price Predictor")
st.markdown("*ML-powered predictions using Random Forest, Gradient Boosting, Linear & Ridge Regression*")

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("Training Period", ["2y", "5y", "10y"], index=1)
days_history = st.sidebar.slider("Price History (days)", 30, 365, 90)

col1, col2 = st.sidebar.columns(2)
predict_btn = col1.button("🔮 Predict", use_container_width=True)
indicators_btn = col2.button("📊 Indicators", use_container_width=True)

# Price History Chart (always shown)
st.subheader(f"📉 Price History — {symbol}")
with st.spinner("Loading price history..."):
    try:
        res = requests.get(f"{API_URL}/history/{symbol}?days={days_history}", timeout=15)
        if res.status_code == 200:
            history = res.json()["history"]
            df = pd.DataFrame(history)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["close"],
                mode="lines", name="Close Price",
                line=dict(color="#1a56a0", width=2)
            ))
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Price (USD)",
                template="plotly_dark", height=350,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not fetch history for {symbol}")
    except Exception as e:
        st.error(f"API not running. Start with: uvicorn main:app --reload")

# Prediction Section
if predict_btn:
    st.subheader(f"🔮 Prediction for {symbol}")
    with st.spinner(f"Training models on {period} of data... this takes ~30 seconds"):
        try:
            res = requests.get(f"{API_URL}/predict/{symbol}?period={period}", timeout=120)
            if res.status_code == 200:
                data = res.json()

                # Main metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"${data['current_price']}")
                c2.metric("Predicted Tomorrow", f"${data['predicted_next_day']}",
                         delta=f"{data['change_percent']}%")
                c3.metric("Best Model", data['best_model'])
                c4.metric("Price Change", f"${data['change']}",
                         delta=f"{data['change_percent']}%")

                st.divider()

                # All model predictions
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**All Model Predictions**")
                    preds = data['all_model_predictions']
                    pred_df = pd.DataFrame({
                        "Model": list(preds.keys()),
                        "Predicted Price ($)": list(preds.values())
                    })
                    fig2 = px.bar(pred_df, x="Model", y="Predicted Price ($)",
                                 color="Predicted Price ($)",
                                 color_continuous_scale="blues",
                                 template="plotly_dark")
                    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig2, use_container_width=True)

                with col_b:
                    st.markdown("**Model Accuracy (Test Set)**")
                    acc = data['model_accuracies']
                    acc_df = pd.DataFrame(acc).T.reset_index()
                    acc_df.columns = ["Model", "RMSE", "MAE", "R²"]
                    st.dataframe(acc_df, use_container_width=True, hide_index=True)

            else:
                st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Try a shorter period like '2y'.")
        except Exception as e:
            st.error(f"Error: {e}")

# Indicators Section
if indicators_btn:
    st.subheader(f"📊 Technical Indicators — {symbol}")
    with st.spinner("Fetching indicators..."):
        try:
            res = requests.get(f"{API_URL}/indicators/{symbol}", timeout=30)
            if res.status_code == 200:
                ind = res.json()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"${ind['current_price']}")
                c2.metric("RSI", ind['RSI'],
                         delta="Overbought" if ind['RSI'] > 70 else ("Oversold" if ind['RSI'] < 30 else "Neutral"))
                c3.metric("MACD", ind['MACD'])
                c4.metric("Volume Ratio", ind['Volume_Ratio'])

                st.divider()
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Moving Averages**")
                    ma_data = {
                        "Indicator": ["SMA 20", "SMA 50", "EMA 20", "BB Upper", "BB Lower"],
                        "Value": [ind['SMA_20'], ind['SMA_50'], ind['EMA_20'],
                                 ind['BB_Upper'], ind['BB_Lower']]
                    }
                    st.dataframe(pd.DataFrame(ma_data), use_container_width=True, hide_index=True)

                with col_b:
                    st.markdown("**Bollinger Band Position**")
                    bb_pos = ind['BB_Position']
                    fig3 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=bb_pos,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "#1a56a0"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "#2ecc71"},
                                {'range': [0.3, 0.7], 'color': "#f39c12"},
                                {'range': [0.7, 1], 'color': "#e74c3c"}
                            ]
                        }
                    ))
                    fig3.update_layout(height=250, template="plotly_dark",
                                      margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("*Built with FastAPI + scikit-learn + Streamlit | Data from Yahoo Finance*")
