import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

# Fix random seed for reproducibility
np.random.seed(42)

# Load Iris dataset
iris = load_iris()
X = iris.data[:, 2:]  # Use petal length and width as input
y = (iris.target != 0).astype(int).reshape(-1, 1)  # Binary classification

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Load the model
model = LogisticRegression()
model.load("classifier2_weights.npz")

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")