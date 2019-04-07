import numpy as np

###### FUNCS ######
def SIGMOID(x, isDerivative=False):
    if isDerivative:
        return x * (1 - x)
    return 1 / (1 + np.e**(-x))
###################

class SimpleLayer():
    def __init__(self, neurons, prev_neurons, activation=SIGMOID):
        self.W = np.random.rand(neurons, prev_neurons) *2-1
        self.B = np.zeros((neurons, 1))
        self.A = None
        self.activation = activation
    def forward_pass(self, x):
        self.A = x
        return self.activation(self.W @ self.A + self.B)
    def update_parameters(self, delta, learning_rate):
        self.W = self.W - delta @ self.A.T * learning_rate
        self.B = self.B - np.mean(delta, axis=1, keepdims=True) * learning_rate

class MemoryLayer(SimpleLayer):
    def __init__(self, neurons, prev_neurons, activation=SIGMOID):
        super().__init__(neurons, prev_neurons, activation)
        self.Wm = np.random.rand(neurons, neurons) *2-1
        self.H = np.zeros_like(self.B)
        self.pH = np.zeros_like(self.H)
    def forward_pass(self, x):
        self.A = x
        self.pH = self.H
        self.H = self.activation(self.W @ self.A + self.Wm @ self.pH + self.B)
        return self.H
    def update_parameters(self, delta, learning_rate):
        super().update_parameters(delta, learning_rate)
        self.Wm = delta @ self.pH.T * learning_rate
