import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from Linear_regression import LinearRegression

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use sepal length and width as input
y = iris.data[:, 2].reshape(-1, 1)  # Predict petal length

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Load the model
model = LinearRegression()
model.load("regression1_weights.npz")

# Evaluate the model
mse = model.score(X_test, y_test)
print(f"Mean Squared Error: {mse}")