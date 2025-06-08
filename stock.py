import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

# ----- Stock Ticker List -----
stock_list = ['INFY.BO', 'TCS.NS', 'RELIANCE.BO', 'GOOGL', 'HDFCBANK.BO']

# ----- Helper Functions -----
def get_stock_data(ticker, period):
    data = yf.download(ticker, period=period)
    return data

def plot_data(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['Close'].rolling(window=20).mean(), label='20-day MA', color='orange')
    plt.title('Stock Price & Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def get_summary(data):
    return data.describe().to_string()

def predict_next_day(data):
    close_data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    x, y = [], []
    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    x = np.array(x)
    y = np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)

    last_60 = scaled_data[-60:].reshape((1, 60, 1))
    prediction = model.predict(last_60)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    return f"Predicted next closing price: ₹{predicted_price:.2f}"

# ----- Gradio Interface -----
def dashboard(ticker, period):
    data = get_stock_data(ticker, period)
    if data.empty:
        return "No data found", None, "N/A"
    plot = plot_data(data)
    summary = get_summary(data)
    prediction = predict_next_day(data)
    return prediction, plot, summary

with gr.Blocks() as demo:
    gr.Markdown("# 📈 AI Stock Market Dashboard")

    with gr.Row():
        ticker_dropdown = gr.Dropdown(choices=stock_list, label="Select a Stock", interactive=True)
        period = gr.Radio(["1mo", "3mo", "6mo", "1y"], label="Select Time Period", value="3mo")
        run_btn = gr.Button("Run Analysis")

    with gr.Tabs():
        with gr.TabItem("📊 Chart + AI Prediction"):
            prediction_output = gr.Textbox(label="AI Price Prediction")
            plot_output = gr.Plot()
        with gr.TabItem("📋 Stock Summary"):
            summary_output = gr.Textbox(label="Statistical Summary", lines=20, max_lines=None)

    run_btn.click(fn=dashboard, inputs=[ticker_dropdown, period], outputs=[prediction_output, plot_output, summary_output])

demo.launch()
