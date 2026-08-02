import pandas as pd
import random
from datetime import datetime, timedelta

# -----------------------------
# Configuration
# -----------------------------
NUM_RECORDS = 2000

products = {
    "Furniture": ["Chair", "Table", "Sofa", "Bookcase", "Desk"],
    "Office Supplies": ["Paper", "Pen", "Binder", "Stapler", "Notebook"],
    "Technology": ["Laptop", "Phone", "Printer", "Monitor", "Keyboard"]
}

regions = {
    "North": ["Delhi", "Chandigarh", "Lucknow"],
    "South": ["Bangalore", "Chennai", "Hyderabad"],
    "East": ["Kolkata", "Bhubaneswar", "Patna"],
    "West": ["Mumbai", "Pune", "Ahmedabad"]
}

rows = []

start_date = datetime(2023,1,1)

for i in range(1, NUM_RECORDS+1):

    category = random.choice(list(products.keys()))
    product = random.choice(products[category])

    region = random.choice(list(regions.keys()))
    city = random.choice(regions[region])

    quantity = random.randint(1,10)

    price = random.randint(100,5000)

    sales = quantity * price

    discount = random.choice([0,5,10,15,20,25,30])

    discount_amount = sales * discount/100

    final_sales = sales - discount_amount

    profit = round(final_sales * random.uniform(0.05,0.35),2)

    order_date = start_date + timedelta(days=random.randint(0,730))

    rows.append([
        f"ORD{i:05}",
        order_date.strftime("%Y-%m-%d"),
        f"CUST{random.randint(1000,9999)}",
        product,
        category,
        product,
        region,
        city,
        quantity,
        round(final_sales,2),
        profit,
        discount
    ])

columns = [
    "Order ID",
    "Order Date",
    "Customer ID",
    "Product",
    "Category",
    "Sub-Category",
    "Region",
    "City",
    "Quantity",
    "Sales",
    "Profit",
    "Discount"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv("retail_sales.csv", index=False)

print("Dataset Generated Successfully!")
print(df.head())

print("\nShape :", df.shape)