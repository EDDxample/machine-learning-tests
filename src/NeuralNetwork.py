import numpy as np

###### FUNCS ######
def COST(Ypredicted, Yreal, isDerivative=False):
    if isDerivative:
        return (Ypredicted - Yreal)
    return np.mean((Yreal - Ypredicted)**2)
###################

class NeuralNetwork():
    def __init__(self, topology):
        self.layers = []
        for current_layer, prev_layer in zip(topology[1:],topology[:-1]):
            self.layers.append(current_layer['type'](current_layer['size'], prev_layer['size']))
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
    def cycle(self, x, count): # Only if x.shape == forward_pass(x).shape
        outputs = [x]
        for i in range(count):
            outputs.append(self.forward_pass(outputs[-1]))
        return outputs

    def __len__(self): return len(self.layers)
    def __getitem__(self, index): return self.layers[index]

