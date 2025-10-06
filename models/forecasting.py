# forecasting.py - Robust Forecasting (auto-detect columns, Prophet + LSTM + fallbacks)
# Usage: python forecasting.py
# Expects a CSV file "processed_sales_data.csv" in the same folder (auto-detects date/product/sales columns)

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- Prophet import (try both names) ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except Exception:
        PROPHET_AVAILABLE = False

# --- ML / LSTM imports ---
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except Exception:
    # If TF/sklearn not available, we'll still run Prophet / simple fallback
    TENSORFLOW_AVAILABLE = False

# -----------------------------
# 1. Load Processed Data
# -----------------------------
file_name = "processed_sales_data.csv"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Could not find {file_name} in current folder. Put your CSV here.")

df = pd.read_csv(file_name)
print("📂 Columns found in dataset:", list(df.columns))

# Helper to find best column from candidate keywords
def find_col_by_keywords(keywords):
    for col in df.columns:
        lname = col.lower()
        for kw in keywords:
            if kw in lname:
                return col
    return None

# Candidate lists (prioritized)
date_col = find_col_by_keywords(["date", "day", "timestamp", "order_date"])
prod_col = find_col_by_keywords(["product_id", "productid", "product", "sku", "item", "item_id"])
sales_col = find_col_by_keywords(["units_sold", "units sold", "sales", "qty", "quantity", "demand", "units_ordered", "unitsordered"])

# If any not found, try looser matching
if date_col is None:
    date_col = find_col_by_keywords(["time", "timestamp"])
if prod_col is None:
    prod_col = find_col_by_keywords(["product code", "code", "name"])
if sales_col is None:
    sales_col = find_col_by_keywords(["sold", "orders", "units"])

# If still missing, show error with helpful message
if not date_col or not prod_col or not sales_col:
    raise ValueError(
        "❌ Could not detect required columns automatically.\n"
        f"Detected columns: {list(df.columns)}\n"
        "Required: a date column, a product identifier column, and a sales/units column.\n"
        "Common names: Date/date, Product_ID/product_id/product, Sales/units_sold/qty/demand.\n"
        "If your columns use different names, rename them in the CSV or re-run with those names."
    )

print(f"✅ Using columns -> date: '{date_col}', product: '{prod_col}', sales: '{sales_col}'")

# Convert types and clean
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
df = df.dropna(subset=[date_col, prod_col, sales_col]).copy()
df = df.sort_values([prod_col, date_col])

# Ensure output folders
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------
# 2. LSTM helpers (if tensorflow available)
# -----------------------------
def train_lstm(series, n_lags=7, epochs=10):
    """Train a simple LSTM on series (pandas Series). Returns (model, scaler)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - n_lags):
        X.append(scaled[i:i+n_lags, 0])
        y.append(scaled[i+n_lags, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([LSTM(50, activation='relu', input_shape=(n_lags, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, scaler

def forecast_lstm(model, scaler, series, steps=30, n_lags=7):
    """Generate `steps` forecasts using trained LSTM and scaler."""
    data = scaler.transform(series.values.reshape(-1, 1)).flatten().tolist()
    preds = []
    for _ in range(steps):
        x_input = np.array(data[-n_lags:]).reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)
        data.append(yhat[0][0]); preds.append(yhat[0][0])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

# -----------------------------
# 3. Fallback forecasting (simple)
# -----------------------------
def simple_moving_average_forecast(series, steps=30, window=7):
    """Simple moving average forecast fallback (no external libs)."""
    if len(series) == 0:
        return np.zeros(steps)
    last_window = series.values[-window:] if len(series) >= window else series.values
    avg = np.mean(last_window)
    return np.array([avg] * steps)

# Robust MAPE helper (avoid div-by-zero)
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

# -----------------------------
# 4. Forecasting Loop
# -----------------------------
forecast_list = []
products = df[prod_col].unique()
print(f"\nFound {len(products)} products. Forecasting each (this may take time).")

for product in products:
    print(f"\n🔄 Processing product: {product}")
    product_df = df[df[prod_col] == product][[date_col, sales_col]].copy()
    product_df = product_df.sort_values(date_col)
    product_df = product_df.set_index(date_col).asfreq('D', method=None)  # keep daily freq; may introduce NaNs
    # fill missing daily sales with forward/backfill or 0 (choose forward then fill 0)
    product_df[sales_col] = product_df[sales_col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Prepare data for Prophet
    prophet_yhat = None
    lstm_yhat = None

    # --- Prophet forecasting (if available) ---
    if PROPHET_AVAILABLE:
        try:
            prophet_df = product_df.reset_index().rename(columns={date_col: 'ds', sales_col: 'y'})[['ds', 'y']]
            # Prophet requires at least a few data points
            if len(prophet_df) >= 4:
                model_p = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model_p.fit(prophet_df)
                future = model_p.make_future_dataframe(periods=30)
                forecast_p = model_p.predict(future)
                prophet_yhat = forecast_p['yhat'][-30:].values
            else:
                print("  ℹ️ Not enough data for Prophet -> skipping Prophet for this product.")
        except Exception as e:
            print(f"  ⚠️ Prophet failed for product {product}: {e}")

    # --- LSTM forecasting (if TensorFlow available and enough data) ---
    if TENSORFLOW_AVAILABLE:
        sales_series = product_df[sales_col].dropna()
        # require enough points to create at least one training sample (n_lags+1)
        n_lags = 7
        if len(sales_series) > n_lags + 1:
            try:
                lstm_model, scaler = train_lstm(sales_series, n_lags=n_lags, epochs=10)
                lstm_yhat = forecast_lstm(lstm_model, scaler, sales_series, steps=30, n_lags=n_lags)
            except Exception as e:
                print(f"  ⚠️ LSTM failed for product {product}: {e}")
                lstm_yhat = None
        else:
            print("  ℹ️ Not enough data for LSTM -> skipping LSTM for this product.")
    else:
        print("  ℹ️ TensorFlow not available: skipping LSTM.")

    # --- If neither model produced results, use simple moving average fallback ---
    if prophet_yhat is None and lstm_yhat is None:
        print("  ⚠️ Neither Prophet nor LSTM available for this product — using simple moving average fallback.")
        fallback = simple_moving_average_forecast(product_df[sales_col].fillna(0), steps=30, window=7)
        best_forecast = fallback
    else:
        # Evaluate available forecasts against most recent actuals
        sales_series_full = product_df[sales_col].dropna()
        actual = sales_series_full[-30:].values if len(sales_series_full) >= 1 else np.array([])

        # compute RMSE for each available forecast (use only overlapping length)
        scores = {}
        if prophet_yhat is not None and actual.size > 0:
            p_pred = np.array(prophet_yhat[:len(actual)])
            try:
                rmse_p = np.sqrt(mean_squared_error(actual, p_pred))
                scores['prophet'] = {'rmse': rmse_p, 'pred': np.array(prophet_yhat)}
            except Exception:
                scores['prophet'] = {'rmse': np.inf, 'pred': np.array(prophet_yhat)}
        if lstm_yhat is not None and actual.size > 0:
            l_pred = np.array(lstm_yhat[:len(actual)])
            try:
                rmse_l = np.sqrt(mean_squared_error(actual, l_pred))
                scores['lstm'] = {'rmse': rmse_l, 'pred': np.array(lstm_yhat)}
            except Exception:
                scores['lstm'] = {'rmse': np.inf, 'pred': np.array(lstm_yhat)}

        # Choose best by smallest RMSE (if both exist). If actual is empty (no overlap), prefer LSTM (if exists) then Prophet.
        if len(scores) == 0:
            # no actuals -> prefer LSTM then Prophet
            if lstm_yhat is not None:
                best_forecast = np.array(lstm_yhat)
            else:
                best_forecast = np.array(prophet_yhat)
        else:
            best_key = min(scores.items(), key=lambda x: x[1]['rmse'])[0]
            best_forecast = scores[best_key]['pred']
            print(f"  Selected model for {product}: {best_key} (RMSE={scores[best_key]['rmse']:.3f})")

    # ensure best_forecast is numpy array of length 30
    best_forecast = np.array(best_forecast).flatten()
    if len(best_forecast) < 30:
        # pad with last value
        pad_val = best_forecast[-1] if best_forecast.size > 0 else 0.0
        best_forecast = np.concatenate([best_forecast, np.repeat(pad_val, 30 - len(best_forecast))])
    elif len(best_forecast) > 30:
        best_forecast = best_forecast[:30]

    forecast_dates = pd.date_range(start=product_df.index.max() + pd.Timedelta(days=1), periods=30)
    temp = pd.DataFrame({
        "date": forecast_dates,
        "forecast_best": best_forecast,
        prod_col: product
    })
    forecast_list.append(temp)

# -----------------------------
# 4. Save & Plot Results
# -----------------------------
if len(forecast_list) == 0:
    raise RuntimeError("No forecasts produced. Check your data and library installations.")

forecast_all = pd.concat(forecast_list, ignore_index=True)
out_path = "data/forecast_results.csv"
forecast_all.to_csv(out_path, index=False)
print(f"\n✅ Forecast results saved in {out_path}")

# Plot sample (first product)
try:
    sample_product = forecast_all[prod_col].unique()[0]
    sample_df = forecast_all[forecast_all[prod_col] == sample_product]
    plt.figure(figsize=(10,4))
    plt.plot(sample_df['date'], sample_df['forecast_best'], marker='o', label='Forecast')
    plt.title(f"Forecast - {sample_product}")
    plt.xlabel("Date"); plt.ylabel("Forecasted units")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not plot sample forecast: {e}")
