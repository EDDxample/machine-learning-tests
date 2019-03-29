import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# CREAR EL DATASET
n = 500 # número de registros en nuestros datos
p = 2   # cuantas características tenemos sobre cada registro

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

# CLASE DE LA CAPA DE LA RED
class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f

        # Vector Bias 1xN de valores (-1, 1)
        self.b = np.random.rand(1, n_neur)      * 2 - 1
        # Vector Weights M x N
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1


# FUNCIONES DE ACTIVACION
sigmoid = (
    lambda x: 1 / (1 + np.e**(-x)),
    lambda x: x * (1 - x)
)

def create_nn(topology, act_f):
    nn = []
    for connections, neurons in zip(topology[:-1], topology[1:]):
        nn.append(neural_layer(connections, neurons, act_f))
    return nn

# NN p -> hidden -> 1
topology = [p, 4, 8, 1]
nn = create_nn(topology, sigmoid)

l2_cost = (
    lambda Yp, Yr: np.mean((Yp - Yr)**2),
    lambda Yp, Yr: (Yp - Yr)
)

def train(nn, test_i, test_o, err_f, l_rate=0.5, train=True):

    out = [(None, test_i)] # [(z0, a0),(z1, a1),...]

    # Forward pass
    for layer in nn:
        z = out[-1][1] @ layer.W + layer.b
        a = layer.act_f[0](z)
        out.append((z, a))
    
    if train:
        # Backward pass
        deltas = []
        for l in reversed(range(0, len(nn))):
            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(nn) - 1:
                # delta de la última capa
                deltas.insert(0, l2_cost[1](a, test_o) * nn[l].act_f[1](a))
            else:
                # delta respecto a capa previa
                deltas.insert(0, deltas[0] @ _W.T * nn[l].act_f[1](a))
            
            _W = nn[l].W
            
            # Gradient descent
            nn[l].b = nn[l].b - np.mean(deltas[0], axis=0, keepdims=True) * l_rate
            nn[l].W = nn[l].W - out[l][1].T @ deltas[0] * l_rate
    return out[-1][1]



# FINAL TEST

import time

topology = [p, 4, 8, 1]
neural_net = create_nn(topology, sigmoid)

loss = []

for i in range(2000):

    pY = train(neural_net, X, Y, l2_cost, l_rate=0.05)

    if i % 25 == 0:

        loss.append(l2_cost[0](pY, Y))

        res = 50

        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)
        _Y = np.zeros((res,res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neural_net, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]
        
        print('prediction...', loss[-1])


        if i == 1975:
            plt.pcolormesh(_x0, _x1, _Y, cmap='coolwarm')
            plt.axis('equal')

            plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c="skyblue")
            plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c='salmon')
            
            plt.show()
            plt.plot(range(len(loss)), loss)
            plt.show()