import numpy as np
from typing import List, Tuple

class Initializer:
    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        raise NotImplementedError("Initializer must implement the __call__ method.")
    
class HeInitializer(Initializer):
    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        return np.random.randn(*shape) * np.sqrt(2. / shape[0])
    
class XavierInitializer(Initializer):
    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        return np.random.randn(*shape) * np.sqrt(1. / shape[0])
    
class ZerosInitializer(Initializer):
    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        return np.zeros(shape)
    
class Activation:
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Activation must implement the __call__ method.")
    
    def backward(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Activation must implement the backward method.")
    
class ReLU(Activation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    def backward(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)
    
class Sigmoid(Activation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z_clipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_clipped))
    
    def backward(self, Z: np.ndarray) -> np.ndarray:
        A = self.__call__(Z)
        return A * (1 - A)
    
class Tanh(Activation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)
    
    def backward(self, Z: np.ndarray) -> np.ndarray:
        A = self.__call__(Z)
        return 1 - A ** 2
    
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
        return np.ones_like(Z)
    
class LossFunction:
    def __call__(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        raise NotImplementedError("LossFunction must implement the __call__ method.")
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LossFunction must implement the backward method.")
    
class MeanSquaredError(LossFunction):
    def __call__(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        return np.mean((Y_pred - Y_true) ** 2)
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        return 2 * (Y_pred - Y_true) / Y_true.size
    
class BinaryCrossEntropy(LossFunction):
    def __call__(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred)) / m
        return float(loss)

    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
        return -(Y_true / Y_pred - (1 - Y_true) / (1 - Y_pred)) / m
    
class CategoricalCrossEntropy(LossFunction):
    def __call__(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(Y_true * np.log(Y_pred)) / m
        return float(loss)

    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        m = Y_true.shape[0]
        return (Y_pred - Y_true) / m
    
class DenseLayer:
    def __init__(self, input_dim: int, output_dim: int, initializer: Initializer, activation: Activation):
        """
        Initializes a dense layer with the given input and output dimensions, initializer, and activation function.
        Args:
            input_dim (int): The number of input features to the layer.
            output_dim (int): The number of output features from the layer.
            initializer (Initializer): An instance of an initializer to initialize the weights of the layer.
            activation (Activation): An instance of an activation function to apply to the layer's output.
            W (np.ndarray): The weight matrix of the layer, initialized using the provided initializer.
            b (np.ndarray): The bias vector of the layer, initialized to zeros.
            grads (dict): A dictionary to store the gradients of the weights and biases during backpropagation.
            cache (dict): A dictionary to store intermediate values during the forward pass for use in backpropagation.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer
        self.activation = activation

        self.W = self.initializer((self.input_dim, self.output_dim))
        self.b = np.zeros((1, self.output_dim))
        
        self.cache = {}
        self.grads = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Z = XW + b
        A = activation(Z)
        """
        self.cache["X"] = X
        self.cache["Z"] = np.dot(X, self.W) + self.b
        self.cache["A"] = self.activation(self.cache["Z"])
        return self.cache["A"]

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        dZ = dA * activation'(Z)
        dW = X^T * dZ
        db = sum(dZ)
        dX_prev = dZ * W^T
        """
        X = self.cache["X"]
        Z = self.cache["Z"]

        dZ = dA * self.activation.backward(Z)
        
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX_prev = np.dot(dZ, self.W.T) # A[l-1]
        
        self.grads["dW"] = dW
        self.grads["db"] = db
        
        return dX_prev
    
    def update_parameters(self, learning_rate: float):
        self.W -= learning_rate * self.grads["dW"]
        self.b -= learning_rate * self.grads["db"]

class Sequential:
    def __init__(self):
        """
        Initializes a Sequential model, which is a linear stack of layers. The model maintains a list of layers, a loss function, and a history of loss values during training.
        Args:
            layers (List[DenseLayer]): A list to store the layers of the model.
            loss_function (LossFunction): The loss function used to compute the cost during training.
            loss_history (list): A list to store the history of loss values during training for analysis and visualization.
        """
        self.layers: List[DenseLayer] = []
        self.loss_function: LossFunction = None
        self.loss_history = []
        self.task: str = "binary"

    def add(self, layer: DenseLayer):
        self.layers.append(layer)

    def compile(self, loss_function: LossFunction, task: str = "binary"):
        """
        Compiles the Sequential model by setting the loss function and the type of task (binary classification, multi-class classification, or regression).
        Args:
            loss_function (LossFunction): An instance of a loss function to compute the cost during training.
            task (str): The type of task the model is being trained for. 
                It can be "binary" for binary classification, 
                "multiclass" for multi-class classification, 
                or "regression" for regression tasks. 
                This information can be used to determine the appropriate activation function and loss function during training and prediction.
        """
        self.loss_function = loss_function
        self.task = task

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray):
        dA = self.loss_function.backward(Y_pred, Y_true)
        
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float, print_every: int = 10):
        for epoch in range(epochs):
            Y_pred = self.forward(X)
            
            loss = self.loss_function(Y_pred, Y)
            self.loss_history.append(loss)
            
            self.backward(Y_pred, Y)

            for layer in self.layers:
                layer.update_parameters(learning_rate)
                
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        outputs = self.forward(X)
        
        if self.task == "binary":
            return (outputs >= threshold).astype(int)
        elif self.task == "multiclass":
            return np.argmax(outputs, axis=1)
        elif self.task == "regression":
            return outputs
        else:
            raise ValueError("Invalid task type. Supported types: 'binary', 'multiclass', 'regression'.")
    
if __name__ == "__main__":
    np.random.seed(42)

    # ========================================================
    # Problem 1: BINARY CLASSIFICATION
    # Data: Classification of points inside/outside a circle (Non-linear)
    # ========================================================
    print("--- 1. BINARY CLASSIFICATION ---")
    X_bin = np.random.randn(400, 2)
    # Labels = 1 if radius > 1.5, otherwise = 0
    Y_bin = (np.sum(X_bin**2, axis=1) > 1.5).astype(int).reshape(-1, 1)

    model_bin = Sequential()
    model_bin.add(DenseLayer(input_dim=2, output_dim=16, initializer=HeInitializer(), activation=ReLU()))
    model_bin.add(DenseLayer(input_dim=16, output_dim=8, initializer=HeInitializer(), activation=ReLU()))
    model_bin.add(DenseLayer(input_dim=8, output_dim=1, initializer=XavierInitializer(), activation=Sigmoid())) # Output Sigmoid

    model_bin.compile(loss_function=BinaryCrossEntropy(), task="binary")
    model_bin.fit(X_bin, Y_bin, epochs=1000, learning_rate=0.1, print_every=200)

    acc_bin = np.mean(model_bin.predict(X_bin) == Y_bin) * 100
    print(f"Accuracy Binary: {acc_bin:.2f}%\n")


    # ========================================================
    # Problem 2: MULTICLASS CLASSIFICATION (Multi-class Classification)
    # Data: 3 clusters of points (blobs) scattered in 3 different corners
    # ========================================================
    print("--- 2. MULTICLASS CLASSIFICATION ---")
    # Create 3 clusters of points
    blob1 = np.random.randn(100, 2) + [2, 2]   # Class 0
    blob2 = np.random.randn(100, 2) + [-2, -2] # Class 1
    blob3 = np.random.randn(100, 2) + [2, -2]  # Class 2
    X_multi = np.vstack([blob1, blob2, blob3])
    
    # Create labels and One-Hot Encoding (Using np.eye technique)
    Y_classes = np.array([0]*100 + [1]*100 + [2]*100) 
    Y_multi = np.eye(3)[Y_classes] # Convert to one-hot matrix (300, 3)

    model_multi = Sequential()
    model_multi.add(DenseLayer(input_dim=2, output_dim=16, initializer=HeInitializer(), activation=ReLU()))
    model_multi.add(DenseLayer(input_dim=16, output_dim=3, initializer=XavierInitializer(), activation=Softmax())) # Output Softmax (3 classes)

    model_multi.compile(loss_function=CategoricalCrossEntropy(), task="multiclass")
    model_multi.fit(X_multi, Y_multi, epochs=1000, learning_rate=0.1, print_every=200)

    # Predict returns the index of the highest class, compare with Y_classes
    preds_multi = model_multi.predict(X_multi)
    acc_multi = np.mean(preds_multi == Y_classes) * 100
    print(f"Accuracy Multiclass: {acc_multi:.2f}%\n")


    # ========================================================
    # Problem 3: REGRESSION (Regression)
    # Data: Non-linear function y = 3x1 - 2x2^2 + 1 + noise
    # ========================================================
    print("--- 3. REGRESSION ---")
    X_reg = np.random.randn(300, 2)
    # Formula for generating Y (with some Gaussian noise)
    Y_reg = (3 * X_reg[:, 0] - 2 * (X_reg[:, 1]**2) + 1 + np.random.randn(300)*0.1).reshape(-1, 1)

    model_reg = Sequential()
    model_reg.add(DenseLayer(input_dim=2, output_dim=32, initializer=HeInitializer(), activation=ReLU()))
    model_reg.add(DenseLayer(input_dim=32, output_dim=16, initializer=HeInitializer(), activation=ReLU()))
    model_reg.add(DenseLayer(input_dim=16, output_dim=1, initializer=XavierInitializer(), activation=Linear())) # Output Linear

    model_reg.compile(loss_function=MeanSquaredError(), task="regression")
    model_reg.fit(X_reg, Y_reg, epochs=1000, learning_rate=0.005, print_every=200)

    preds_reg = model_reg.predict(X_reg)
    mae_reg = np.mean(np.abs(preds_reg - Y_reg))
    print(f"Mean Absolute Error (MAE) Regression: {mae_reg:.4f}")