import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except Exception:
        PROPHET_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

file_name = "processed_sales_data.csv"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Could not find {file_name} in current folder.")

df = pd.read_csv(file_name)
print("📂 Columns found in dataset:", list(df.columns))

def find_col_by_keywords(keywords):
    for col in df.columns:
        lname = col.lower()
        for kw in keywords:
            if kw in lname:
                return col
    return None

date_col = find_col_by_keywords(["date", "day"])
prod_col = find_col_by_keywords(["product", "sku", "item"])
sales_col = find_col_by_keywords(["sales", "units_sold", "qty", "demand", "quantity"])

if not date_col or not prod_col or not sales_col:
    raise ValueError("❌ Could not detect required columns. Please ensure dataset has Date, Product_ID, and Sales/Qty.")

print(f"✅ Using columns -> date: '{date_col}', product: '{prod_col}', sales: '{sales_col}'")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
df = df.dropna(subset=[date_col, prod_col, sales_col]).copy()

def train_lstm(series, n_lags=7, epochs=10):
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
    data = scaler.transform(series.values.reshape(-1, 1)).flatten().tolist()
    preds = []
    for _ in range(steps):
        x_input = np.array(data[-n_lags:]).reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)
        data.append(yhat[0][0]); preds.append(yhat[0][0])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

products = df[prod_col].unique()
print(f"\nFound {len(products)} products. Forecasting each...\n")

for product in products:
    print(f"🔄 Forecasting for product: {product}")
    product_df = df[df[prod_col] == product][[date_col, sales_col]].sort_values(date_col)

    prophet_yhat = None
    if PROPHET_AVAILABLE and len(product_df) > 4:
        prophet_df = product_df.rename(columns={date_col: "ds", sales_col: "y"})
        model_p = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model_p.fit(prophet_df)
        future = model_p.make_future_dataframe(periods=30)
        forecast_p = model_p.predict(future)
        prophet_yhat = forecast_p['yhat'][-30:].values

    lstm_yhat = None
    if TENSORFLOW_AVAILABLE and len(product_df) > 8:
        sales_series = product_df.set_index(date_col)[sales_col]
        train_series = sales_series.iloc[:int(len(sales_series)*0.8)]
        lstm_model, scaler = train_lstm(train_series)
        lstm_yhat = forecast_lstm(lstm_model, scaler, sales_series, steps=30)

    if lstm_yhat is not None:
        best_forecast = lstm_yhat
        chosen = "LSTM"
    elif prophet_yhat is not None:
        best_forecast = prophet_yhat
        chosen = "Prophet"
    else:
        avg = product_df[sales_col].mean()
        best_forecast = np.array([avg]*30)
        chosen = "Fallback (Average)"

    forecast_dates = pd.date_range(start=product_df[date_col].max() + pd.Timedelta(days=1), periods=30)

    print(f"✅ Selected model: {chosen}")
    print(f"{'Date':<12} {'Forecast':<10}")
    for d, f in zip(forecast_dates, best_forecast):
        print(f"{d.date()}  {f:.2f}")
    print("-"*40)

    plt.figure(figsize=(10,4))
    plt.plot(product_df[date_col], product_df[sales_col], label="Actual Sales")
    plt.plot(forecast_dates, best_forecast, label="Forecast", marker="o")
    plt.title(f"Sales Forecast for {product} ({chosen})")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n🎯 Forecasting complete. All results printed and plotted.")
