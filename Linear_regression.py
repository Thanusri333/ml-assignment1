import numpy as np

class LinearRegression:
    def __init__(self):
        self.W = None  # Weights
        self.b = None  # Bias
        self.loss_history = []  # To store loss during training

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, learning_rate=0.01):
        """
        Train the linear regression model using gradient descent with early stopping.
        """
        np.random.seed(42)  # Fix random seed for reproducibility

        n_samples, n_features = X.shape
        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        y = y.reshape(-1, n_outputs)

        # Initialize weights and bias
        self.W = np.random.randn(n_features, n_outputs)
        self.b = np.zeros((1, n_outputs))

        # Split data into training and validation sets
        val_size = int(0.1 * n_samples)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        best_loss = float('inf')
        best_W, best_b = self.W, self.b
        patience_count = 0

        for epoch in range(max_epochs):
            # Shuffle training data with a fixed seed
            indices = np.arange(X_train.shape[0])
            np.random.seed(epoch)  # Fix seed for shuffling
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            # Mini-batch gradient descent
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                y_pred = X_batch @ self.W + self.b

                # Compute loss (MSE + L2 regularization)
                mse = np.mean((y_pred - y_batch) ** 2)
                reg_loss = regularization * np.sum(self.W ** 2)
                loss = mse + reg_loss

                # Backward pass
                dW = (X_batch.T @ (y_pred - y_batch)) / batch_size + 2 * regularization * self.W
                db = np.mean(y_pred - y_batch, axis=0, keepdims=True)

                # Update weights and bias
                self.W -= learning_rate * dW
                self.b -= learning_rate * db

            # Validation loss
            y_val_pred = self.predict(X_val)
            val_loss = np.mean((y_val_pred - y_val) ** 2)
            self.loss_history.append(val_loss)  # Append validation loss to history

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_W, best_b = self.W, self.b
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        # Set the best weights and bias
        self.W, self.b = best_W, best_b

    def predict(self, X):
        """
        Predict the output for the given input data.
        """
        return X @ self.W + self.b

    def score(self, X, y):
        """
        Compute the mean squared error for the given input data and target values.
        """
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)

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