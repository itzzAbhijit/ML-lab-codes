import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

print("\nTesting Data:")
print(test_data.head())

print("Training Data:")
print(train_data.head())

X_train = train_data[['x']]
y_train = train_data['y']

X_test = test_data[['x']]
y_test = test_data['y']

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error (MSE):", mse)
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nRegression Equation:")
print("y =", model.coef_[0], "* x +", model.intercept_)

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

sorted_indices = X_test['x'].argsort()

plt.plot(
X_test['x'].iloc[sorted_indices],
y_pred[sorted_indices],
color='red',
linewidth=2,
label='Regression Line'
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
