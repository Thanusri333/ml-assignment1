import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Linear_regression import LinearRegression

# Fix random seed for reproducibility
np.random.seed(42)

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use sepal length and width as input
y = iris.data[:, 2].reshape(-1, 1)  # Predict petal length

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train, batch_size=32, max_epochs=100, patience=3)

# Save the model
model.save("regression1_weights.npz")

# Plot training loss
plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Training Loss Over Epochs")
plt.savefig("regression1_loss.png")
plt.show()