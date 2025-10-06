# forecasting.py - Milestone 2: Forecasting (Prophet + LSTM + Evaluation)
 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from prophet import Prophet

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pickle, os, warnings

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler
 
warnings.filterwarnings("ignore")
 
# -----------------------------

# 1. Load Cleaned Data

# -----------------------------

df = pd.read_csv("processed_sales_data.csv")

date_col = [c for c in df.columns if "date" in c.lower()][0]

df[date_col] = pd.to_datetime(df[date_col])
 
os.makedirs("models", exist_ok=True)

os.makedirs("data", exist_ok=True)
 
# -----------------------------

# 2. Helper — LSTM

# -----------------------------

def train_lstm(series, n_lags=7, epochs=10):

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []

    for i in range(len(scaled) - n_lags):

        X.append(scaled[i:i+n_lags, 0]); y.append(scaled[i+n_lags, 0])

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
 
# -----------------------------

# 3. Forecasting Loop

# -----------------------------

forecast_list = []

for product in df['Product ID'].unique():

    print(f"🔄 Training Prophet & LSTM for {product}...")

    product_df = df[df['Product ID'] == product][[date_col, 'Sales']]
 
    # Prophet

    prophet_df = product_df.rename(columns={date_col: 'ds', 'Sales': 'y'})

    model_p = Prophet(yearly_seasonality=True, weekly_seasonality=True)

    model_p.fit(prophet_df)

    future = model_p.make_future_dataframe(periods=30)

    forecast_p = model_p.predict(future)

    yhat_p = forecast_p['yhat'][-30:]
 
    # LSTM

    sales_series = product_df.set_index(date_col)['Sales']

    train_series = sales_series.iloc[:int(len(sales_series)*0.8)]

    lstm_model, scaler = train_lstm(train_series)

    yhat_l = forecast_lstm(lstm_model, scaler, sales_series, steps=30)
 
    # Eval (includes MAPE)

    actual = sales_series[-30:] if len(sales_series) >= 30 else sales_series

    def mape(y_true,y_pred): return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
 
    mae_p, rmse_p, mape_p = mean_absolute_error(actual,yhat_p[:len(actual)]), np.sqrt(mean_squared_error(actual,yhat_p[:len(actual)])), mape(actual,yhat_p[:len(actual)])

    mae_l, rmse_l, mape_l = mean_absolute_error(actual,yhat_l[:len(actual)]), np.sqrt(mean_squared_error(actual,yhat_l[:len(actual)])), mape(actual,yhat_l[:len(actual)])
 
    print(f"Prophet MAE:{mae_p:.2f} RMSE:{rmse_p:.2f} MAPE:{mape_p:.1f}%")

    print(f"LSTM    MAE:{mae_l:.2f} RMSE:{rmse_l:.2f} MAPE:{mape_l:.1f}%")
 
    best_forecast = yhat_l if rmse_l < rmse_p else yhat_p

    forecast_dates = pd.date_range(start=product_df[date_col].max()+pd.Timedelta(days=1), periods=30)

    temp = pd.DataFrame({"date":forecast_dates,"forecast_best":best_forecast,"Product ID":product})

    forecast_list.append(temp)
 
forecast_all = pd.concat(forecast_list)

forecast_all.to_csv("data/forecast_results.csv", index=False)

print("\n✅ Forecast results saved in data/forecast_results.csv")
 
# Sample plot

sample = df['Product ID'].unique()[0]

sample_df = forecast_all[forecast_all['Product ID']==sample]

plt.plot(sample_df['date'], sample_df['forecast_best'], label="Forecast")

plt.title(f"Forecast - {sample}"); plt.legend(); plt.show()

 