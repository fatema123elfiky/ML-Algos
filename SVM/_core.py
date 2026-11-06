import numpy as np


class SVM :
    def __init__(self,lambda_param=0.01, learning_rate=0.01, n_iterations=1000):
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)  
        n_samples, n_features = X.shape


        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.b
        return np.sign(linear_output)

    
    def get_cost(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        hinge = np.maximum(0, 1 - y_ * (np.dot(X, self.weights) + self.b))
        return self.lambda_param * np.dot(self.weights,self.weights) + np.mean(hinge)

