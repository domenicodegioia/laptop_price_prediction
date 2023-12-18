import pandas as pd

df = pd.read_csv('laptop_price.csv', encoding="latin-1")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# data pre-processing
df = df.drop("laptop_ID", axis=1)

df = df.drop("Product", axis=1)

df = df.join(pd.get_dummies(df.Company))
df = df.drop("Company", axis=1)

df = df.join(pd.get_dummies(df.TypeName))
df = df.drop("TypeName", axis=1)

df["ScreenResolution"] = df.ScreenResolution.str.split(" ").apply(lambda x: x[-1])
df["Screen Width"] = df.ScreenResolution.str.split("x").apply(lambda x: x[0]).astype("int")
df["Screen Height"] = df.ScreenResolution.str.split("x").apply(lambda x: x[1]).astype("int")
df = df.drop("ScreenResolution", axis=1)

df["CPU Brand"] = df.Cpu.str.split(" ").apply(lambda x: x[0])
cpu_gategories = pd.get_dummies(df["CPU Brand"])
cpu_gategories.columns = [col + "_CPU" for col in cpu_gategories.columns]
df = df.join(cpu_gategories)
df = df.drop("CPU Brand", axis=1)
df["CPU Frequency"] = df.Cpu.str.split(" ").apply(lambda x: x[-1])
df = df.drop("Cpu", axis=1)
df["CPU Frequency"] = df["CPU Frequency"].str[:-3].astype("float")

df["Ram"] = df["Ram"].str[:-2].astype("int")

df["Memory Amount"] = df.Memory.str.split(" ").apply(lambda x: x[0])
df["Memory Type"] = df.Memory.str.split(" ") .apply(lambda x: x[1])
df = df.join(pd.get_dummies(df["Memory Type"]))
df = df.drop("Memory Type", axis=1)
df = df.drop("Memory", axis=1)

def turn_memory_into_MB(value):
    if "GB" in value:
        return float(value[:value.find("GB")]) * 1000
    elif "TB" in value:
        return float(value[:value.find("TB")]) * 1000000

df["Memory Amount"] = df["Memory Amount"].apply(turn_memory_into_MB)

df["Weight"] = df["Weight"].str[:-2].astype("float")

df["GPU Brand"] = df.Gpu.str.split(" ").apply(lambda x: x[0])
gpu_gategories = pd.get_dummies(df["GPU Brand"])
gpu_gategories.columns = [col + "_GPU" for col in gpu_gategories.columns]
df = df.join(gpu_gategories)
df = df.drop("GPU Brand", axis=1)
df = df.drop("Gpu", axis=1)

df = df.join(pd.get_dummies(df.OpSys))
df = df.drop("OpSys", axis=1)



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18,15))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()

target_correlations = df.corr()["Price_euros"].apply(abs).sort_values()
selected_features = target_correlations[:-21].index
selected_features = list(selected_features)
selected_features.append("Price_euros")

limited_df = df[selected_features]
plt.figure(figsize=(18,15))
sns.heatmap(limited_df.corr(), annot=True, cmap="YlGnBu")
plt.show()



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X, y = limited_df.drop("Price_euros", axis=1), limited_df["Price_euros"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

forest = RandomForestRegressor()
forest.fit(X_train, y_train)
forest.score(X_test, y_test)

y_pred = forest.predict(X_test)

plt.figure(figsize=(12,8))
plt.scatter(y_pred, y_test)
plt.plot(range(0,6000), range(0,6000), c="red")
plt.show()