import numpy as np, matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork, COST
from sklearn.datasets import make_circles


dataset_input, dataset_output = make_circles(n_samples=500, factor=0.5, noise=0.05)
dataset_input = dataset_input.T                  # fix (500,2) to (2,500)
dataset_output = dataset_output[:, np.newaxis].T # fix (500, ) to (1,500)

topology = [dataset_input.shape[0], 8, dataset_output.shape[0]]
nn = NeuralNetwork(topology)

errors = []

cycles = 2000

for i in range(cycles):
    # predict + train
    prediction = nn.backprop(dataset_input, dataset_output, learing_rate=0.05)
    
    if i % 25:
        errors.append(COST(prediction, dataset_output))
    if i == cycles - 1:
        plt.plot(range(len(errors)), errors)
        plt.show()