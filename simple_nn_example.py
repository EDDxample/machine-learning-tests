import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_circles

from src.NeuralNetwork import NeuralNetwork, COST
from src.Layers import SimpleLayer

dataset_input, dataset_output = make_circles(n_samples=500, factor=0.5, noise=0.05)
dataset_input = dataset_input.T                  # fix (500,2) to (2,500)
dataset_output = dataset_output[:, np.newaxis].T # fix (500, ) to (1,500)

topology = [
    {'type':SimpleLayer, 'size':dataset_input.shape[0]},
    {'type':SimpleLayer, 'size':7},
    {'type':SimpleLayer, 'size':dataset_output.shape[0]}
]

nn = NeuralNetwork(topology)

errors = []

cycles = 2000

for i in range(cycles):
    # predict + train
    prediction = nn.backprop(dataset_input, dataset_output, learning_rate=0.05)
    
    if i % 25:
        errors.append(COST(prediction, dataset_output))
    if i == cycles - 1:

        res = 70
        _x = np.linspace(-1.5, 1.5, res)
        _y = np.linspace(-1.5, 1.5, res)
        _Y = np.zeros((res,res))

        for i, _x_ in enumerate(_x):
            for j, _y_ in enumerate(_y):
                 _Y[i, j] = nn.forward_pass(np.array([[_x_], [_y_]]))

        plt.pcolormesh(_x, _y, _Y, cmap='coolwarm')
        plt.axis('equal')

        plt.scatter(dataset_input[0, dataset_output[0,:] == 0], dataset_input[1, dataset_output[0,:] == 0], c='skyblue')
        plt.scatter(dataset_input[0, dataset_output[0,:] == 1], dataset_input[1, dataset_output[0,:] == 1], c='salmon')
        plt.show()

        plt.plot(range(len(errors)), errors)
        plt.show()