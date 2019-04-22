import numpy as np

###### FUNCS ######
def SIGMOID(x, isDerivative=False):
    if isDerivative:
        return x * (1 - x)
    return 1 / (1 + np.e**(-x))
def COST(Ypredicted, Yreal, isDerivative=False):
    if isDerivative:
        return (Ypredicted - Yreal)
    return np.mean((Yreal - Ypredicted)**2)
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

class NeuralNetwork():
    def __init__(self, topology):
        self.layers = []
        for current_layer, prev_layer in zip(topology[1:],topology[:-1]):
            self.layers.append(SimpleLayer(current_layer, prev_layer))
    def forward_pass(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward_pass(a)
        return a
    def backprop(self, xr, yr, learning_rate=0.5, cost_func=COST):
        output = self.forward_pass(xr)
        flag = True
        for layer in reversed(self):
            if flag: # last layer
                flag = False
                delta = cost_func(output, yr, True) * layer.activation(output, True)
            else:
                delta = _W @ delta * _A
            # saves
            _W = layer.W.T
            _A = layer.activation(layer.A, True)

            layer.update_parameters(delta, learning_rate)

        return output
    
    def __len__(self): return len(self.layers)
    def __getitem__(self, index): return self.layers[index]