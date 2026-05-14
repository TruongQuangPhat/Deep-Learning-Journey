import numpy as np
from typing import Tuple, List
from abc import ABC, abstractmethod

class Initializer(ABC):
    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        pass

class HeInitializer(Initializer):
    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        return np.random.randn(*shape) * np.sqrt(2. / shape[0])
    
class XavierInitializer(Initializer):
    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        return np.random.randn(*shape) * np.sqrt(1. / shape[0])
    
class Activation(ABC):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, Z: np.ndarray) -> np.ndarray:
        pass

class ReLU(Activation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    def backward(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

class LeakyReLU(Activation):
    def __call__(self, Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(Z > 0, Z, alpha * Z)

    def backward(self, Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(Z > 0, 1, alpha)
    
class Tanh(Activation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)
    
    def backward(self, Z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(Z) ** 2
    
class Sigmoid(Activation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z_clipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_clipped))

    def backward(self, Z: np.ndarray) -> np.ndarray:
        A = self(Z)
        return A * (1 - A)
    
class Linear(Activation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return Z
    
    def backward(self, Z: np.ndarray) -> np.ndarray:
        return np.ones_like(Z)
    
class Softmax(Activation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def backward(self, Z: np.ndarray) -> np.ndarray:
        return np.ones_like(Z)  # Placeholder, actual backward pass for softmax is handled with cross-entropy loss
    
class LossFunction(ABC):
    @abstractmethod
    def __call__(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        pass

class BinaryCrossEntropy(LossFunction):
    def __call__(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-8, 1 - 1e-8)  # Avoid log(0)
        loss = -np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred)) / m
        return loss
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-8, 1 - 1e-8)  # Avoid division by zero
        return (Y_pred - Y_true) / (Y_pred * (1 - Y_pred) * m) # dL/dA
    
class MeanSquaredError(LossFunction):
    def __call__(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        return np.mean((Y_pred - Y_true) ** 2)
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        return 2 * (Y_pred - Y_true) / Y_true.size

class CategoricalCrossEntropy(LossFunction):
    def __call__(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-8, 1 - 1e-8)  # Avoid log(0)
        loss = -np.sum(Y_true * np.log(Y_pred)) / m
        return loss
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-8, 1 - 1e-8)  # Avoid division by zero
        return (Y_pred - Y_true) / m # dL/dA (Softmax + Cross-Entropy)
    
class Regularization(ABC):
    @abstractmethod
    def __call__(self, W: np.ndarray, m: int) -> float:
        pass
    
    @abstractmethod
    def backward(self, W: np.ndarray, m: int) -> np.ndarray:
        pass

class L2Regularization(Regularization):
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_
    
    def __call__(self, W: np.ndarray, m: int) -> float:
        return (self.lambda_ / (2 * m)) * np.sum(W ** 2)
    
    def backward(self, W: np.ndarray, m: int) -> np.ndarray:
        return (self.lambda_ / m) * W

class L1Regularization(Regularization):
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_
    
    def __call__(self, W: np.ndarray, m: int) -> float:
        return (self.lambda_ / m) * np.sum(np.abs(W))
    
    def backward(self, W: np.ndarray, m: int) -> np.ndarray:
        return (self.lambda_ / m) * np.sign(W)

class Dropout:
    def __init__(self, keep_prob: float):
        if not (0 < keep_prob <= 1):
            raise ValueError("keep_prob must be in the range (0, 1].")
        self.keep_prob = keep_prob
        self.mask = None

    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            self.mask = None
            return X

        self.mask = np.random.binomial(1, self.keep_prob, size=X.shape) / self.keep_prob
        return X * self.mask
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("Dropout backward called before training forward pass.")
        return dA * self.mask

class DenseLayer:
    def __init__(self, input_dim: int, output_dim: int,
                 initializer: Initializer = HeInitializer(),
                 activation: Activation = ReLU(),
                 regularization: Regularization = None,
                 dropout: Dropout = None):
        """
        Initialize a dense layer.
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            initializer (Initializer): Weight initializer.
            activation (Activation): Activation function.
            regularization (Regularization): Regularization method.
            dropout (Dropout): Dropout layer for regularization.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer
        self.activation = activation
        self.regularization = regularization
        self.dropout = dropout
        
        self.W = self.initializer((input_dim, output_dim))
        self.b = np.zeros((1, output_dim))

        self.cache = {} # save intermediate values for backward pass
        self.grad = {} # save gradients for weights and biases

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache["X"] = X 

        Z = np.dot(X, self.W) + self.b
        self.cache["Z"] = Z

        A = self.activation(Z)
        
        if self.dropout is not None:
            A = self.dropout(A, training)
        self.cache["A"] = A

        return A
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        A = self.cache["A"]
        Z = self.cache["Z"]
        X = self.cache["X"]

        if self.dropout is not None:
            dA = self.dropout.backward(dA)

        dZ = dA * self.activation.backward(Z)

        m = X.shape[0]
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)

        if self.regularization is not None:
            dW += self.regularization.backward(self.W, m)

        self.grad["dW"] = dW
        self.grad["db"] = db

        return dX

    def update_params(self, learning_rate: float):
        self.W -= learning_rate * self.grad["dW"]
        self.b -= learning_rate * self.grad["db"]

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.loss_history = []
        self.task = "binary"

    def add(self, layer: DenseLayer):
        self.layers.append(layer)

    def compile(self, loss_function: LossFunction, task: str = "binary"):
        self.loss_function = loss_function
        self.task = task

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A, training)
        return A

    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray):
        dA = self.loss_function.backward(Y_pred, Y_true)

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def compute_loss(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        m = Y_true.shape[0]

        loss = self.loss_function(Y_pred, Y_true)
        for layer in self.layers:
            if layer.regularization is not None:
                loss += layer.regularization(layer.W, m)
        return loss

    def update_params(self, learning_rate: float):
        for layer in self.layers:
            layer.update_params(learning_rate)
    
    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float, print_every: int = 10):
        if self.loss_function is None:
            raise RuntimeError("Model must be compiled before training. Call compile() first.")
        
        for epoch in range(epochs):
            Y_pred = self.forward(X, training=True)

            loss = self.compute_loss(Y_pred, Y)
            self.loss_history.append(loss)

            self.backward(Y_pred, Y)

            self.update_params(learning_rate)

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self.loss_function is None:
            raise RuntimeError("Model must be compiled before prediction. Call compile() first.")
        
        Y_pred = self.forward(X, training=False)

        if self.task == "binary":
            return (Y_pred >= threshold).astype(int)
        elif self.task == "multiclass":
            return np.argmax(Y_pred, axis=1)
        elif self.task == "regression":
            return Y_pred
        else:
            raise ValueError("Unsupported task type. Use 'binary', 'multiclass', or 'regression'.")


if __name__ == "__main__":
    np.random.seed(42)

    # ========================================================
    # Problem 1: BINARY CLASSIFICATION with L2 + DROPOUT
    # Data: Classification of points inside/outside a circle (Non-linear)
    # ========================================================
    print("--- 1. BINARY CLASSIFICATION with L2 + DROPOUT ---")
    X_bin = np.random.randn(400, 2)
    # Labels = 1 if radius > 1.5, otherwise = 0
    Y_bin = (np.sum(X_bin**2, axis=1) > 1.5).astype(int).reshape(-1, 1)

    model_bin = Sequential()
    model_bin.add(DenseLayer(input_dim=2, output_dim=32, initializer=HeInitializer(), 
                             activation=ReLU(), regularization=L2Regularization(lambda_=0.001),
                             dropout=Dropout(keep_prob=0.8)))
    model_bin.add(DenseLayer(input_dim=32, output_dim=16, initializer=HeInitializer(), 
                             activation=ReLU(), regularization=L2Regularization(lambda_=0.001),
                             dropout=Dropout(keep_prob=0.8)))
    model_bin.add(DenseLayer(input_dim=16, output_dim=8, initializer=HeInitializer(), 
                             activation=ReLU(), regularization=L2Regularization(lambda_=0.001),
                             dropout=Dropout(keep_prob=0.8)))
    model_bin.add(DenseLayer(input_dim=8, output_dim=1, initializer=XavierInitializer(), 
                             activation=Sigmoid()))

    model_bin.compile(loss_function=BinaryCrossEntropy(), task="binary")
    model_bin.fit(X_bin, Y_bin, epochs=10000, learning_rate=0.1, print_every=200)

    acc_bin = np.mean(model_bin.predict(X_bin) == Y_bin) * 100
    print(f"Accuracy Binary: {acc_bin:.2f}%\n")


    # ========================================================
    # Problem 2: MULTICLASS CLASSIFICATION with L2 + DROPOUT
    # Data: 3 clusters of points (blobs) scattered in 3 different corners
    # ========================================================
    print("--- 2. MULTICLASS CLASSIFICATION with L2 + DROPOUT ---")
    # Create 3 clusters of points
    blob1 = np.random.randn(100, 2) + [2, 2]   # Class 0
    blob2 = np.random.randn(100, 2) + [-2, -2] # Class 1
    blob3 = np.random.randn(100, 2) + [2, -2]  # Class 2
    X_multi = np.vstack([blob1, blob2, blob3])
    
    # Create labels and One-Hot Encoding (Using np.eye technique)
    Y_classes = np.array([0]*100 + [1]*100 + [2]*100) 
    Y_multi = np.eye(3)[Y_classes] # Convert to one-hot matrix (300, 3)

    model_multi = Sequential()
    model_multi.add(DenseLayer(input_dim=2, output_dim=32, initializer=HeInitializer(), 
                               activation=ReLU(), regularization=L2Regularization(lambda_=0.001),
                               dropout=Dropout(keep_prob=0.8)))
    model_multi.add(DenseLayer(input_dim=32, output_dim=16, initializer=HeInitializer(), 
                               activation=ReLU(), regularization=L2Regularization(lambda_=0.001),
                               dropout=Dropout(keep_prob=0.8)))
    model_multi.add(DenseLayer(input_dim=16, output_dim=3, initializer=XavierInitializer(), 
                               activation=Softmax()))

    model_multi.compile(loss_function=CategoricalCrossEntropy(), task="multiclass")
    model_multi.fit(X_multi, Y_multi, epochs=10000, learning_rate=0.1, print_every=200)

    # Predict returns the index of the highest class, compare with Y_classes
    preds_multi = model_multi.predict(X_multi)
    acc_multi = np.mean(preds_multi == Y_classes) * 100
    print(f"Accuracy Multiclass: {acc_multi:.2f}%\n")


    # ========================================================
    # Problem 3: REGRESSION with L2 + DROPOUT
    # Data: Non-linear function y = 3x1 - 2x2^2 + 1 + noise
    # ========================================================
    print("--- 3. REGRESSION with L2 + DROPOUT ---")
    X_reg = np.random.randn(300, 2)
    # Formula for generating Y (with some Gaussian noise)
    Y_reg = (3 * X_reg[:, 0] - 2 * (X_reg[:, 1]**2) + 1 + np.random.randn(300)*0.1).reshape(-1, 1)

    model_reg = Sequential()
    model_reg.add(DenseLayer(input_dim=2, output_dim=32, initializer=HeInitializer(), 
                             activation=ReLU(), regularization=L2Regularization(lambda_=0.0001),
                             dropout=Dropout(keep_prob=0.95)))
    model_reg.add(DenseLayer(input_dim=32, output_dim=16, initializer=HeInitializer(), 
                             activation=ReLU(), regularization=L2Regularization(lambda_=0.0001),
                             dropout=Dropout(keep_prob=0.95)))
    model_reg.add(DenseLayer(input_dim=16, output_dim=1, initializer=XavierInitializer(), 
                             activation=Linear()))

    model_reg.compile(loss_function=MeanSquaredError(), task="regression")
    model_reg.fit(X_reg, Y_reg, epochs=10000, learning_rate=0.01, print_every=200)

    preds_reg = model_reg.predict(X_reg)
    mae_reg = np.mean(np.abs(preds_reg - Y_reg))
    print(f"Mean Absolute Error (MAE) Regression: {mae_reg:.4f}")
    