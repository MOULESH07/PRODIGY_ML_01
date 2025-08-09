# Task 1: Linear Regression - House Prices


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset

# Upload 'train.csv' file from Kaggle dataset
from google.colab import files
uploaded = files.upload()

# Read the CSV file
df = pd.read_csv("train.csv")

# Show first few rows
print(df.head())


# 2. Select Features

# We use square footage (GrLivArea), Bedrooms (BedroomAbvGr), Bathrooms (FullBath)
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

X = df[features]
y = df[target]


# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)


# 5. Predictions

y_pred = model.predict(X_test)


# 6. Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# 7. Visualization

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
