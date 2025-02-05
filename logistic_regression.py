import numpy as np

class LogisticRegression:
    def __init__(self):
        self.W = None  # Weights
        self.b = None  # Bias
        self.loss_history = []  # To store loss during training

    def fit(self, X, y, learning_rate=0.01, max_epochs=100, batch_size=32):
        """
        Train the logistic regression model using gradient descent.
        """
        np.random.seed(42)  # Fix random seed for reproducibility

        n_samples, n_features = X.shape
        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        y = y.reshape(-1, n_outputs)

        # Initialize weights and bias
        self.W = np.random.randn(n_features, n_outputs)
        self.b = np.zeros((1, n_outputs))

        for epoch in range(max_epochs):
            # Shuffle data with a fixed seed
            indices = np.arange(n_samples)
            np.random.seed(epoch)  # Fix seed for shuffling
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Forward pass
                logits = X_batch @ self.W + self.b
                y_pred = 1 / (1 + np.exp(-logits))

                # Compute loss (log loss)
                loss = -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
                self.loss_history.append(loss)

                # Backward pass
                dW = (X_batch.T @ (y_pred - y_batch)) / batch_size
                db = np.mean(y_pred - y_batch, axis=0, keepdims=True)

                # Update weights and bias
                self.W -= learning_rate * dW
                self.b -= learning_rate * db

    def predict(self, X):
        """
        Predict the class labels for the given input data.
        """
        logits = X @ self.W + self.b
        y_pred = 1 / (1 + np.exp(-logits))
        return (y_pred > 0.5).astype(int)

    def score(self, X, y):
        """
        Compute the accuracy for the given input data and target values.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def save(self, filepath):
        """
        Save the model parameters to a file.
        """
        np.savez(filepath, W=self.W, b=self.b)

    def load(self, filepath):
        """
        Load the model parameters from a file.
        """
        data = np.load(filepath)
        self.W, self.b = data['W'], data['b']