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

class Layer():
    def __init__(self, n_neurons, n_prev_neurons, activation=SIGMOID):
        self.W = np.random.rand(n_neurons, n_prev_neurons) *2-1
        self.B = np.random.rand(n_neurons, 1)              *2-1
        
        self.activation = activation

class NeuralNetwork():
    def __init__(self, topology):
        self.layers = []
        for neurons, prev_neurons in zip(topology[1:],topology[:-1]):
            self.layers.append(Layer(neurons, prev_neurons))
    
    def forward_pass(self, input, isTraining=False):
        activatedLayers = [input]

        for layer in self.layers:
            z = layer.W @ activatedLayers[-1] + layer.B
            a = layer.activation(z)
            activatedLayers.append(a)
        
        if isTraining:
            return activatedLayers
        return activatedLayers[-1]

    def train(self, test_input, test_output, learing_rate=0.5, cost_function=COST):
        pass

    def __len__(self): return len(self.layers)
    def __getitem__(self, index): return self.layers[index]

