import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display the first five rows
print("First Five Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check the number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# Clean the dataset

# Remove duplicate rows
df = df.drop_duplicates()

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values after conversion
print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Display cleaned dataset
print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nData Cleaning Completed!")
# Customer Churn Analysis

total_customers = len(df)

churned_customers = (df["Churn"] == "Yes").sum()

retained_customers = (df["Churn"] == "No").sum()

churn_rate = (churned_customers / total_customers) * 100

retention_rate = (retained_customers / total_customers) * 100

print("\nCUSTOMER CHURN ANALYSIS")

print("Total Customers:", total_customers)

print("Churned Customers:", churned_customers)

print("Retained Customers:", retained_customers)

print("Churn Rate:", round(churn_rate, 2), "%")

print("Retention Rate:", round(retention_rate, 2), "%")
# Churn Analysis by Contract

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Contract:")
print(contract_churn.round(2))

# Plot the results
contract_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()

plt.savefig("churn_by_contract.png", dpi=300)
plt.show()
# Churn Analysis by Internet Service

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Internet Service:")
print(internet_churn.round(2))

# Plot the results
internet_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()

plt.savefig("churn_by_internet_service.png", dpi=300)
plt.show()
# Monthly Charges Analysis

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig("monthly_charges_vs_churn.png", dpi=300)
plt.show()

# Compare average monthly charges
print("\nAverage Monthly Charges by Churn Status:")

print(
    df.groupby("Churn")["MonthlyCharges"]
    .mean()
    .round(2)
)
# Customer Tenure Analysis

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=20,
    multiple="stack"
)

plt.title("Customer Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("customer_tenure_analysis.png", dpi=300)
plt.show()

# Average tenure
print("\nAverage Customer Tenure:")

print(
    df.groupby("Churn")["tenure"]
    .mean()
    .round(2)
)
# Churn Analysis by Payment Method

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Payment Method:")
print(payment_churn.round(2))

payment_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5)
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Churn")
plt.tight_layout()

plt.savefig("churn_by_payment_method.png", dpi=300)
plt.show()
# Historical Revenue Proxy

df["HistoricalRevenueProxy"] = (
    df["MonthlyCharges"] * df["tenure"]
)

print("\nAverage Historical Revenue Proxy by Churn:")

print(
    df.groupby("Churn")["HistoricalRevenueProxy"]
    .mean()
    .round(2)
)

# Plot the comparison
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="HistoricalRevenueProxy"
)

plt.title("Historical Revenue Proxy by Churn Status")
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges × Tenure")

plt.tight_layout()

plt.savefig("historical_revenue_proxy.png", dpi=300)
plt.show()
# Final Summary

print("\n" + "=" * 40)
print("CUSTOMER RETENTION ANALYSIS SUMMARY")
print("=" * 40)

print("Total Customers:", len(df))

print("Churned Customers:", (df["Churn"] == "Yes").sum())

print(
    "Overall Churn Rate:",
    round((df["Churn"] == "Yes").mean() * 100, 2),
    "%"
)

print(
    "Overall Retention Rate:",
    round((df["Churn"] == "No").mean() * 100, 2),
    "%"
)

print(
    "Average Tenure:",
    round(df["tenure"].mean(), 2),
    "months"
)

print(
    "Average Monthly Charges:",
    round(df["MonthlyCharges"].mean(), 2)
)

print("\nContract Churn Percentages:")
print(contract_churn.round(2))

print("\nInternet Service Churn Percentages:")
print(internet_churn.round(2))

print("\nPayment Method Churn Percentages:")
print(payment_churn.round(2))

print("\nAnalysis Completed Successfully!")
