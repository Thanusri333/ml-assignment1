import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

# Fix random seed for reproducibility
np.random.seed(42)

# Load Iris dataset
iris = load_iris()
X = iris.data  # Use all features as input
y = (iris.target != 0).astype(int).reshape(-1, 1)  # Binary classification

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train, learning_rate=0.01, max_epochs=100, batch_size=32)

# Save the model
model.save("classifier3_weights.npz")