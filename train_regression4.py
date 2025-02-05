import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Linear_regression import LinearRegression

# Load Iris dataset
iris = load_iris()
X = iris.data[:, 1:3]  # Use sepal width and petal length as input
y = iris.data[:, 3].reshape(-1, 1)  # Predict petal width

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train, batch_size=32, max_epochs=100, patience=3)

# Save the model
model.save("regression4_weights.npz")

# Plot training loss
plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("regression4_loss.png")
plt.show()