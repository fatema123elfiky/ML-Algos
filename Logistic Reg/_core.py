import numpy as np


class LogisticReg :


    def __init__(self,weights,b ,learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = weights
        self.b = b

    def sigmoid (self, X):

        return 1 / (1 + np.exp(-(np.dot(self.weights,X)+self.b)))

    def cost_function(self, X, y):

        cost = 0.0
        for i in range(X.shape[0]):
            f = self.sigmoid(X[i])
            cost += y[i] * np.log(f) + (1 - y[i]) * np.log(1 - f)
        cost = cost / (-X.shape[0])
        return cost

    def compute_dervatives(self, X, y, w, b):
        m,n = X.shape
        dj_dw = np.zeros(n)
        dj_db = 0.0
        for i in range(m):
            f = self.sigmoid(X[i])
            for j in range(n):
                dj_dw[j] += (f - y[i]) * X[i][j]
            dj_db += (f - y[i])
        return dj_db / m, dj_dw / m

    
    def gradient_descent(self, X, y):

        

        for i in range(self.num_iterations):
            dj_db, dj_dw = self.compute_dervatives(X, y, self.weights, self.b)
            self.weights = self.weights - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db 

        return self.weights, self.b       



    def fit(self , X ,Y):
        self.gradient_descent(X, Y)

    def predict(self , X):
        predictions = [] 
        for x in X:
            pred = self.sigmoid(x)
            if pred >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
                 
        return np.array(predictions)