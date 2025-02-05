import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Fix random seed for reproducibility
np.random.seed(42)

# Load Iris dataset
iris = load_iris()
X = iris.data[:, 2:]  # Use petal length and width as input
y = (iris.target != 0).astype(int).reshape(-1, 1)  # Binary classification

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train, learning_rate=0.01, max_epochs=100, batch_size=32)

# Save the model
model.save("classifier2_weights.npz")

# Plot decision regions
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train.ravel(), clf=model, legend=2)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Decision Regions (Classifier 2)")
plt.savefig("classifier2_decision_regions.png")
plt.show()