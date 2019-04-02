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

        for layer in self:
            z = layer.W @ activatedLayers[-1] + layer.B
            a = layer.activation(z)
            activatedLayers.append(a)
        
        if isTraining:
            return activatedLayers
        return activatedLayers[-1]
    
    def backprop(self, dataset_input, dataset_output, learing_rate=0.5, cost_func=COST):
        layer_outputs = self.forward_pass(dataset_input, True)

        deltas = [] # layer errors

        for L in reversed(range(len(self))): # from layers.length-1 to 0
            a = layer_outputs[L + 1]
            a_prev = layer_outputs[L]
            activ_a = self[L].activation(a, True)
            
            if L == len(self) - 1:
                # last layer's error
                deltas.insert(0, cost_func(a, dataset_output, True) * activ_a)
            else:
                # previous layer's error
                deltas.insert(0, nextW @ deltas[0] * activ_a)
            
            nextW = self[L].W.T

            # fix layer components
            self[L].B -= np.mean(deltas[0], axis=1, keepdims=True) * learing_rate
            self[L].W -= deltas[0] @ a_prev.T * learing_rate
        

    def __len__(self): return len(self.layers)
    def __getitem__(self, index): return self.layers[index]

