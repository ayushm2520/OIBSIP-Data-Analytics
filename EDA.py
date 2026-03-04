import pandas as pd ;
import numpy as np ;
import matplotlib.pyplot as plt ;
  
df=pd.read_csv("NAICS.csv")
print(df)
print(df.describe())
print(df.columns)

#calculation of basic stats
#conversion into numeric 
df["2017 NAICS US Code"]=pd.to_numeric(df["2017 NAICS US Code"],errors="coerce")
print("Mean   : " ,df["2017 NAICS US Code"].mean())
print("Median : " ,df["2017 NAICS US Code"].median())
print("Mode   : " ,df["2017 NAICS US Code"].mode()[0])
print("Std Dev: " ,df["2017 NAICS US Code"].std())

print(df["2017 NAICS US Title"].value_counts()) #frequency of store types

top_stores = df["2017 NAICS US Title"].value_counts().head(20) #top retail sector
print(top_stores)


top_stores = df["2017 NAICS US Title"].value_counts().head(20).plot(kind="bar") #visualize store distributon (Bar)
plt.title("Top 20 store Types")
plt.ylabel("count")
plt.show()

code_counts = df["2017 NAICS US Code"].value_counts().sort_index()
plt.figure(figsize=(10,5))
code_counts.head(20).plot(kind="line",marker="o")
plt.title("NAICS code distribution trend")
plt.xlabel("NAICS Code")
plt.ylabel("Frequency")
plt.show()

