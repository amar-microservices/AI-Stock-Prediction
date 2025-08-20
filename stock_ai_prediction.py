# stock_app.py
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="StockPredictor 3000", layout="wide")
st.title("📈 Nifty 50 Stock Prediction")
st.write("This app predicts the next day's **direction** (UP/DOWN) for Nifty 50 stocks. "
         "It's a learning tool, not financial advice. **Do not trade based on this.**")

# --- Nifty 50 Stocks List ---
nifty_50_stocks = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "INFOSYS": "INFY.NS",
    "HUL": "HINDUNILVR.NS",
    "SBIN": "SBIN.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "KOTAK MAHINDRA BANK": "KOTAKBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
    "L&T": "LT.NS",
    "HCL TECHNOLOGIES": "HCLTECH.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "TITAN": "TITAN.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "HINDALCO": "HINDALCO.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "POWERGRID": "POWERGRID.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "M&M": "M&M.NS",
    "WIPRO": "WIPRO.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "JSW STEEL": "JSWSTEEL.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "INDUSIND BANK": "INDUSINDBK.NS",
    "NESTLE": "NESTLEIND.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "GRASIM": "GRASIM.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "DR. REDDY'S": "DRREDDY.NS",
    "DIVIS LAB": "DIVISLAB.NS",
    "HDFC LIFE": "HDFCLIFE.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS",
    "APOLLO HOSPITALS": "APOLLOHOSP.NS",
    "SBILIFE": "SBILIFE.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS",
    "COAL INDIA": "COALINDIA.NS",
    "EICHER MOTORS": "EICHERMOT.NS",
    "UPL": "UPL.NS",
    "HERO MOTOCORP": "HEROMOTOCO.NS",
    "HIND PETRO": "HINDPETRO.NS",
    "BPCL": "BPCL.NS",
    "IOC": "IOC.NS",
    "GAIL": "GAIL.NS"
}

# --- Sidebar for User Input ---
st.sidebar.header("User Input")

# Dropdown for Nifty 50 stocks
selected_stock = st.sidebar.selectbox(
    "Select Nifty 50 Stock:",
    options=list(nifty_50_stocks.keys()),
    index=0  # Default to RELIANCE
)

# Get the ticker symbol for the selected stock
ticker = nifty_50_stocks[selected_stock]

years = st.sidebar.slider("Years of Historical Data:", 1, 10, 5)
train_size = st.sidebar.slider("Training Data Size (%):", 50, 90, 80) / 100

# --- Helper Function to Calculate RSI ---
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Function to get next trading day ---
def get_next_trading_day(last_date):
    # Convert to datetime if it's not already
    if isinstance(last_date, str):
        last_date = datetime.strptime(last_date, "%Y-%m-%d")
    
    # Add one day
    next_day = last_date + timedelta(days=1)
    
    # If it's Saturday, add 2 more days to get to Monday
    if next_day.weekday() == 5:  # Saturday
        next_day += timedelta(days=2)
    # If it's Sunday, add 1 more day to get to Monday
    elif next_day.weekday() == 6:  # Sunday
        next_day += timedelta(days=1)
    
    return next_day

# --- Main Function to Get Data and Train Model ---
def main():
    # 1. Download Data
    data = yf.download(ticker, period=f"{years}y")
    if data.empty:
        st.error("Could not download data. Check your ticker symbol.")
        return

    # Get the last available date in the data
    last_available_date = data.index[-1].strftime("%Y-%m-%d")
    next_trading_day = get_next_trading_day(last_available_date).strftime("%Y-%m-%d")
    
    # Display date information
    st.sidebar.info(f"**Data last updated:** {last_available_date}")
    st.sidebar.info(f"**Predicting for:** {next_trading_day}")

    # 2. Create DataFrame with all necessary columns
    df = data[['Open', 'High', 'Low', 'Close']].copy()
    
    # 3. Create Features (Technical Indicators)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    df['SMA_200'] = df['Close'].rolling(window=200).mean()  # 200-day SMA
    df['RSI'] = compute_rsi(df['Close'])  # Relative Strength Index
    
    # 4. Create Target: 1 if price goes UP tomorrow, 0 if DOWN.
    # Use numpy arrays to avoid pandas alignment issues
    close_prices = df['Close'].values
    tomorrow_close = np.roll(close_prices, -1)  # Shift all values up by one
    target = (tomorrow_close > close_prices).astype(int)
    
    # Add target to DataFrame
    df['Target'] = target
    
    # Remove the last row which has an invalid target (no tomorrow data)
    df = df.iloc[:-1]

    # 5. Drop rows with missing values (from indicator calculations)
    df = df.dropna()

    # 6. Split into Features (X) and Target (y)
    features = ['Close', 'SMA_50', 'SMA_200', 'RSI']
    X = df[features]
    y = df['Target']

    # 7. Split into Train and Test sets
    split_index = int(len(X) * train_size)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # 8. Train the Model
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
    model.fit(X_train, y_train)

    # 9. Make Predictions and Evaluate
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)  # This is our "Win Rate"
    accuracy = model.score(X_test, y_test)

    # 10. Display Results
    st.subheader(f"Model Performance for {selected_stock} ({ticker})")
    st.write(f"**Prediction Date:** {next_trading_day}")
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision (Win Rate for 'UP' Predictions)", f"{precision:.2%}")

    # Show Confusion Matrix using Plotly
    st.write("**Confusion Matrix:** How did the model do on the test data?")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['DOWN', 'UP'],
                    y=['DOWN', 'UP'],
                    text_auto=True)
    st.plotly_chart(fig)

    # 11. Make a prediction for TOMORROW
    latest_data = X.iloc[-1:]  # Get the most recent day's data
    prediction = model.predict(latest_data)[0]
    prediction_proba = model.predict_proba(latest_data)[0]  # Get the confidence

    st.subheader(f"Prediction for {next_trading_day}:")
    direction = "🟢 UP" if prediction == 1 else "🔴 DOWN"
    confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

    col1, col2 = st.columns(2)
    col1.metric("Predicted Direction", direction)
    col2.metric("Model Confidence", f"{confidence:.2%}")

    # 12. Show the historical price chart
    st.subheader("Historical Price Data")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'], 
                    high=df['High'],
                    low=df['Low'], 
                    close=df['Close'],
                    name='OHLC'))
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=f"{selected_stock} ({ticker}) Historical Price")
    st.plotly_chart(fig, use_container_width=True)

    # 13. Show additional charts
    st.subheader("Technical Indicators")
    
    # Create subplots for indicators
    fig = go.Figure()
    
    # Price and Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='50-Day SMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='200-Day SMA', line=dict(color='red')))
    
    fig.update_layout(title="Price and Moving Averages",
                     xaxis_title="Date",
                     yaxis_title="Price (INR)")
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI Chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
    fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig2.update_layout(title="Relative Strength Index (RSI)",
                      xaxis_title="Date",
                      yaxis_title="RSI Value",
                      yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

    # 14. Show the raw data if user wants
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(df.tail(10))

if __name__ == '__main__':
    main()