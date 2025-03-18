import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from yahooquery import search
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib

matplotlib.use("Agg")

app = Flask(__name__)
CORS(app)


def get_stock_ticker(stock_name):
    result = search(stock_name)
    quotes = result.get("quotes", [])
    if quotes:
        return quotes[0]["symbol"]
    return None


def prepare_data(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1))
    X, y = [], []
    for i in range(60, len(prices_scaled)):
        X.append(prices_scaled[i - 60 : i, 0])
        y.append(prices_scaled[i, 0])
    return np.array(X), np.array(y), scaler


def train_model(X, y):
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Define the improved model
    model = Sequential(
        [
            Bidirectional(
                LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.01)),
                input_shape=(X.shape[1], 1),
            ),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(
                LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.01))
            ),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(units=50, activation="relu"),
            Dense(units=1, activation="linear"),
        ]
    )

    # Compile the model with an optimized learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    # Callbacks for training
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )

    # Train the model
    model.fit(
        X,
        y,
        epochs=50,
        batch_size=16,
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
    )

    return model


# def evaluate_model(y, predicted_prices, scaler):
#     y_test = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
#     y_pred = scaler.inverse_transform(predicted_prices.reshape(-1, 1)).flatten()

#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     mape = (abs((y_test - y_pred) / y_test)).mean() * 100
#     r2 = r2_score(y_test, y_pred)
#     accuracy = 100 - mape

#     error_diff = y_pred - y_test

#     return {
#         "mae": mae,
#         "mse": mse,
#         "rmse": rmse,
#         "mape": mape,
#         "r2": r2,
#         "accuracy": accuracy,
#         "mean_prediction_error": error_diff.mean(),
#         "max_error": error_diff.max(),
#         "min_error": error_diff.min(),
#     }


def generate_graph(actual, predicted, dates):
    if dates.empty:
        return None
    plt.figure(figsize=(10, 4))
    plt.plot(dates, actual, label="Actual Prices", color="blue")
    plt.plot(dates, predicted, label="Predicted Prices", color="red")
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    start_dt = pd.to_datetime("2004-01-01")
    end_dt = pd.to_datetime(dates.iloc[-1]) if not dates.empty else start_dt
    plt.xlim(start_dt, end_dt)
    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode("utf-8")
    return img_base64


def generate_graph_2022_2026(actual, predicted, dates):
    if dates.empty:
        return None
    plt.figure(figsize=(10, 4))
    plt.plot(dates, actual, label="Actual Prices", color="blue")
    plt.plot(dates, predicted, label="Predicted Prices", color="red")
    plt.legend()
    plt.title("Zoomed (2022-2026)")
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    start_dt = pd.to_datetime("2022-01-01")
    end_dt = pd.to_datetime("2025-06-01")
    plt.xlim(start_dt, end_dt)
    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode("utf-8")
    return img_base64


@app.route("/api/stock", methods=["GET"])
def get_stock_data():
    stock_name = request.args.get("name")
    start_date = request.args.get("start_date", "2004-01-01")
    end_date = request.args.get("end_date", "2025-01-01")

    if not stock_name:
        return jsonify({"error": "Stock name is required"}), 400

    ticker = get_stock_ticker(stock_name)
    if not ticker:
        return jsonify({"error": "Invalid stock name"}), 404

    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    if not isinstance(hist, pd.DataFrame) or hist.empty:
        return jsonify({"error": "No data found or unexpected format"}), 404

    hist = hist.reset_index()
    if "Close" in hist.columns:
        prices = hist["Close"].dropna().tolist()
    elif "Adj Close" in hist.columns:
        prices = hist["Adj Close"].dropna().tolist()
    else:
        return jsonify({"error": "No valid price data available"}), 404

    X, y, scaler = prepare_data(prices)
    model = train_model(X, y)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    dates = pd.to_datetime(hist["Date"]).iloc[60:]
    main_graph = generate_graph(prices[60:], predictions, dates)
    future_graph = generate_graph_2022_2026(prices[60:], predictions, dates)
    # evaluation_metrics = evaluate_model(y, predictions, scaler)

    return jsonify(
        {
            "main_image": main_graph,
            "future_image": future_graph,
            # "evaluation": evaluation_metrics,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
