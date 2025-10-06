# sales_app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sales Data Analyzer", layout="wide")

st.title("📊 Sales Data Analyzer")
st.markdown("Upload your sales CSV file and explore trends, seasonality, and features.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your sales_data_large.csv", type=["csv"])
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])

    st.write("### Raw Data")
    st.dataframe(df.head())

    # --- Data Cleaning ---
    df['sales'] = df['sales'].fillna(df['sales'].median())
    df = df.drop_duplicates()
    df = df[df['sales'] >= 0].copy()

    # Missing values report
    st.write("### Missing Values (%)")
    st.write((df.isnull().mean() * 100).round(2))

    # --- Feature Engineering ---
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
    df['promotion_flag'] = df['promotion'].apply(lambda x: 1 if str(x).lower() == "yes" else 0)
    df['lag_1'] = df['sales'].shift(1)
    df['rolling_mean_7'] = df['sales'].rolling(window=7).mean().fillna(df['sales'].mean())

    # --- Plots ---
    st.write("### Monthly Sales Trend")
    sales_trend = df.groupby(df['date'].dt.to_period("M"))['sales'].sum()
    sales_trend.index = sales_trend.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10,5))
    sales_trend.plot(kind="line", marker="o", ax=ax, title="Monthly Sales Trend")
    st.pyplot(fig)

    st.write("### Sales Outlier Detection")
    fig, ax = plt.subplots()
    ax.boxplot(df['sales'])
    ax.set_title("Sales Outlier Detection")
    st.pyplot(fig)

    st.write("### Seasonality: Sales by Month")
    monthly_sales = df.groupby(['year','month'])['sales'].sum().unstack(0)
    fig, ax = plt.subplots(figsize=(10,6))
    monthly_sales.plot(marker='o', ax=ax, title="Seasonality: Sales by Month")
    st.pyplot(fig)

    # --- Download Processed CSV ---
    st.write("### Download Processed Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Processed CSV", csv, "processed_sales_data.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to start analyzing.")
