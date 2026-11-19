import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Fuel_consumption_data_set.csv")

X = df[[
    "ENGINESIZE",
    "CYLINDERS",
    "FUELCONSUMPTION_CITY",
    "FUELCONSUMPTION_HWY",
    "FUELCONSUMPTION_COMB"
]]
y = df["CO2EMISSIONS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emission")
plt.title("Actual vs Predicted CO2 Emission")
plt.grid(True)
plt.show()
