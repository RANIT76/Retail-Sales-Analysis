# ===========================================
# Retail Sales & Profit Analysis Project
# ===========================================

# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("Libraries Imported Successfully!")

# ===========================================
# Load Dataset
# ===========================================

df = pd.read_csv("retail_sales.csv")

print("\nDataset Loaded Successfully!")

# First 5 Rows
print(df.head())

# Last 5 Rows
print(df.tail())

# Shape
print("\nShape :", df.shape)

# Columns
print("\nColumns")
print(df.columns)

# Data Types
print("\nData Types")
print(df.dtypes)

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate Records
print("\nDuplicate Records :", df.duplicated().sum())

# Statistics
print("\nStatistics")
print(df.describe())


# ===========================================
# Data Cleaning
# ===========================================

print("\nChecking Missing Values...")
print(df.isnull().sum())

print("\nChecking Duplicate Records...")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

print("\nDataset Shape After Cleaning:", df.shape)

# ===========================================
# Sales by Category
# ===========================================

category_sales = df.groupby("Category")["Sales"].sum()

print("\nSales by Category")
print(category_sales)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
category_sales.plot(kind='bar')

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.show()


# ===========================================
# Sales by Region
# ===========================================

region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(8,5))
region_sales.plot(kind="bar", color="green")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()

# ===========================================
# Profit by Category
# ===========================================

profit_category = df.groupby("Category")["Profit"].sum()

plt.figure(figsize=(8,5))
profit_category.plot(kind="bar", color="orange")
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.show()

# ===========================================
# Sales Distribution
# ===========================================

plt.figure(figsize=(8,5))
plt.hist(df["Sales"], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# ===========================================
# Discount Distribution
# ===========================================

plt.figure(figsize=(8,5))
plt.hist(df["Discount"], bins=10)
plt.title("Discount Distribution")
plt.xlabel("Discount")
plt.ylabel("Frequency")
plt.show()


# ===========================================
# Machine Learning - Sales Prediction
# ===========================================

print("\n========== Machine Learning ==========")

# Features
X = df[["Quantity", "Discount"]]

# Target
y = df["Sales"]

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nModel Training Completed!")

# Accuracy
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

# Predict New Data

quantity = 5
discount = 10

prediction = model.predict([[quantity, discount]])

print("\nPredicted Sales for")
print("Quantity :", quantity)
print("Discount :", discount)

print("Predicted Sales = ₹", round(prediction[0], 2))


# ===========================================
# Pie Chart - Sales by Category
# ===========================================

category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(7,7))
plt.pie(category_sales,
        labels=category_sales.index,
        autopct="%1.1f%%",
        startangle=90)

plt.title("Sales by Category")
plt.show()

# ===========================================
# Pie Chart - Sales by Region
# ===========================================

region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(7,7))
plt.pie(region_sales,
        labels=region_sales.index,
        autopct="%1.1f%%",
        startangle=90)

plt.title("Sales by Region")
plt.show()


# ===========================================
# Monthly Sales Analysis
# ===========================================

# Convert Order Date to Date Format
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create Month Column
df["Month"] = df["Order Date"].dt.month_name()

# Monthly Sales
monthly_sales = df.groupby("Month")["Sales"].sum()

print("\nMonthly Sales")
print(monthly_sales)

# Line Chart

plt.figure(figsize=(12,6))
plt.plot(monthly_sales.index,
         monthly_sales.values,
         marker="o",
         linewidth=3)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.grid(True)

plt.show()


# ===========================================
# Top 10 Products
# ===========================================

top_products = df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(10)

print("\nTop 10 Products")
print(top_products)

plt.figure(figsize=(10,5))
top_products.plot(kind="bar")

plt.title("Top 10 Products")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.show()


# ===========================================
# Correlation Matrix
# ===========================================

correlation = df[["Quantity", "Sales", "Profit", "Discount"]].corr()

print("\nCorrelation Matrix")
print(correlation)

plt.figure(figsize=(6,5))
plt.imshow(correlation, cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Matrix")

plt.show()


# ===========================================
# Correlation Matrix
# ===========================================

correlation = df[["Quantity", "Sales", "Profit", "Discount"]].corr()

print("\nCorrelation Matrix")
print(correlation)

plt.figure(figsize=(6,5))

plt.imshow(correlation, cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Matrix")

plt.show()