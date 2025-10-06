# milestone1_preprocessing_eda.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

print("📂 Current Working Directory:", os.getcwd())
print("📂 Files in Current Directory:", os.listdir())

try:
    df = pd.read_csv("sales_data.csv")
    print("✅ Data Loaded Successfully")
except FileNotFoundError:
    print("❌ Error: sales_data.csv not found in this folder.")
    exit()

print("Data Shape:", df.shape)
print(df.head(), "\n")

print("🔍 Missing Values (%):")
print(df.isnull().mean() * 100, "\n")

# Fill missing values if any
df.fillna({
    "units_sold": df["units_sold"].median(),
    "units_ordered": df["units_ordered"].median(),
    "price": df["price"].mean()
}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.day_name()

# Create holiday/promotion flags (if not already categorical)
df["promotion_flag"] = (df["promotion"] == "Yes").astype(int)
df["holiday_flag"] = (df["holiday"] == "Yes").astype(int)

print("📊 Generating EDA plots...")

pdf = PdfPages("EDA_Report.pdf")

plt.figure(figsize=(10, 5))
df.groupby('date')["units_sold"].sum().plot()
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.grid(True)
plt.savefig("sales_trend.png")
pdf.savefig()  # save to PDF
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df["units_sold"])
plt.title("Units Sold Outliers")
plt.savefig("outliers.png")
pdf.savefig()
plt.show()

plt.figure(figsize=(8, 4))
df.groupby("month")["units_sold"].sum().plot(kind="bar")
plt.title("Monthly Seasonality")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.savefig("seasonality.png")
pdf.savefig()
plt.show()

plt.figure(figsize=(8, 5))
df.groupby("category")["units_sold"].sum().sort_values().plot(kind="barh")
plt.title("Category-wise Sales")
plt.xlabel("Units Sold")
plt.ylabel("Category")
plt.savefig("category_sales.png")
pdf.savefig()
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x="promotion", y="units_sold", data=df, estimator=np.mean)
plt.title("Impact of Promotions on Sales")
plt.savefig("promotion_impact.png")
pdf.savefig()
plt.show()

pdf.close()


df.to_csv("processed_sales_data.csv", index=False)

print("\n📌 Summary")
print(f"Data Quality: {100 - df.isnull().mean().mean() * 100:.2f}% complete")
print("Features Created: month, day_of_week, promotion_flag, holiday_flag")
print("Plots Generated: Saved as PNG + EDA_Report.pdf")
print("✅ Data Preprocessing & EDA Completed.")
