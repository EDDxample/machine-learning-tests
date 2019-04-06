import numpy as np

###### FUNCS ######
def SIGMOID(x, isDerivative=False):
    if isDerivative:
        return x * (1 - x)
    return 1 / (1 + np.e**(-x))
###################

class SimpleLayer():
    def __init__(self, n_neurons, n_prev_neurons, activation=SIGMOID):
        self.W = np.random.rand(n_neurons, n_prev_neurons) *2-1
        self.B = np.random.rand(n_neurons, 1)              *2-1
        self.A = None
        self.activation = activation
    def forward_pass(self, layer_input):
        self.A = layer_input
        return self.activation(self.W @ self.A + self.B)
    def gradient_descent(self, delta, learning_rate):
        self.W = self.W - delta @ self.A.T * learning_rate
        self.B = self.B - np.mean(delta, axis=1, keepdims=True) * learning_rate

class MemoryLayer(SimpleLayer):
    def __init__(self, n_neurons, n_prev_neurons, activation=SIGMOID):
        super().__init__(n_neurons, n_prev_neurons, activation)
        self.Wm = np.random.rand(n_neurons, n_neurons) *2-1
        self.last_pass = np.zeros_like(self.B)
    
    def forward_pass(self, layer_input):
        self.A = layer_input
        self.last_pass = self.activation(self.W @ self.A + self.Wm @ self.last_pass + self.B)
        return self.last_pass

    def gradient_descent(self, delta, learning_rate):
        super().gradient_descent(delta, learning_rate)
        self.Wm = delta @ self.last_pass.T * learning_rate

