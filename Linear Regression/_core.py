import numpy as np
import copy

class LinearRegression :

    def __init__(self,weights,b, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights =weights # w
        self.b = b # b
        


    def cost_function(self, X, y):
        
        cost = 0.0
        for i in range(X.shape[0]):
           f = np.dot(X[i], self.weights) + self.b
           cost  += (f-y[i])**2

        cost = cost / (2 * X.shape[0])
        return cost

    def compute_dervatives(self, X, y,w,b):
        m,n= X.shape
        dj_w = np.zeros(n)
        dj_b = 0.0 

        for i in range(m):
            f = np.dot(X[i], w) + b
            for  j in range(n):
                dj_w[j] += (f-y[i])*X[i][j]
            dj_b += (f-y[i])

        dj_w = dj_w / m
        dj_b = dj_b / m
        return dj_b , dj_w     



    def gradient_descent_o(self, X, y ,w_in , b_in):

       self.weights = copy.deepcopy(w_in)
       self.b = b_in
       for i in range(self.n_iterations):
          dj_db , dj_dw = self.compute_dervatives(X, y, self.weights, self.b)

          self.weights = self.weights - self.learning_rate * dj_dw
          self.b = self.b - self.learning_rate * dj_db

       return self.weights, self.b


    def fit(self, X, y):
        self.gradient_descent_o(X, y, self.weights, self.b)

    def predict(self,X):
        predictions = [ np.dot(x, self.weights) + self.b for x in X]
        return np.array(predictions)          