import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load Data
df = pd.read_csv("AB_NYC_2019.csv")

print("Initial Shape:", df.shape)
print(df.head())

# 1. Data Integrity Check
print("\n--- Data Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Statistical Summary ---")
print(df.describe())

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# ===============================
# 2. Missing Data Handling
# ===============================

# Numeric columns → fill with mean
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing after treatment:")
print(df.isnull().sum())

# ===============================
# 3. Duplicate Removal
# ===============================
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]

print(f"\nDuplicates removed: {before - after}")

# ===============================
# 4. Standardization
# ===============================

# lowercase and strip text columns
for col in cat_cols:
    df[col] = df[col].str.lower().str.strip()


# ===============================
# 5. Outlier Detection (IQR)
# ===============================

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[(data[column] >= lower) & (data[column] <= upper)]

df_clean = df.copy()

for col in num_cols:
    df_clean = remove_outliers_iqr(df_clean, col)

print("\nShape after outlier removal:", df_clean.shape)

# ===============================
# Visualization with matplotlib
# ===============================

# Correlation heatmap using matplotlib
plt.figure(figsize=(6,4))
plt.imshow(df_clean.corr(), interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(df_clean.corr())), df_clean.corr().columns, rotation=90)
plt.yticks(range(len(df_clean.corr())), df_clean.corr().columns)
plt.title("Correlation Heatmap")
plt.show()

# Boxplot for first numeric column
plt.figure(figsize=(6,4))
plt.boxplot(df_clean[num_cols[0]])
plt.title(f"Boxplot of {num_cols[0]}")
plt.show()