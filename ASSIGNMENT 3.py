import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load data
train_data = pd.read_csv("house_train.csv")
test_data = pd.read_csv("house_test.csv")

print("Training Data Shape:", train_data.shape)
print("Testing Data Shape:", test_data.shape)
print(train_data.isnull().sum())

# Remove duplicates
train_data = train_data.drop_duplicates()

# Fill missing values in numerical columns with median
numerical_columns = train_data.select_dtypes(include=np.number).columns
for column in numerical_columns:
    train_data[column] = train_data[column].fillna(train_data[column].median())

# Fill missing values in categorical columns with mode
categorical_columns = train_data.select_dtypes(exclude=np.number).columns
for column in categorical_columns:
    if not train_data[column].mode().empty:
        train_data[column] = train_data[column].fillna(train_data[column].mode()[0])

# Outlier removal on Price using IQR
Q1 = train_data["Price"].quantile(0.25)
Q3 = train_data["Price"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

train_data = train_data[
    (train_data["Price"] >= lower_limit) & (train_data["Price"] <= upper_limit)
]

# Features and target
X = train_data.drop(columns=["Price", "ID"])
y = train_data["Price"]

numerical_features = [
    "Lot_Width",
    "Lot_Depth",
    "Year_Sold",
    "Year_Built",
    "Living_Area",
    "Num_Rooms"
]
categorical_features = ["Location", "Paint_Color"]

# Preprocessing & Polynomial Regression Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False))
            ]),
            numerical_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train, predict, and evaluate
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print("R^2 Score:", r2)
print("MSE:", mse)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
