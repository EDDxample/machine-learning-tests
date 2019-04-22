import numpy as np

text = open('from-scratch/recurrent-nn/end.txt', encoding='utf-8').read()

char_list = set(text)
io_size = len(char_list)

# Translators
char_to_int = {c:i for i, c in enumerate(char_list)}
int_to_char = {i:c for i, c in enumerate(char_list)}

# Hyper-paramenters
h_size = 100
sequence_length = 25
learning_rate = 0.1

# Model parameters
Wxh = np.random.randn(h_size, io_size) * 0.01
Whh = np.random.randn(h_size, h_size)  * 0.01
Why = np.random.randn(io_size, h_size) * 0.01
Bh  = np.zeros((h_size, 1))
By  = np.zeros((io_size, 1))

def backprop(X, Y, H): # X and Y = index lists
    x, h, y, probs = {}, {}, {}, {}

    h[-1] = np.copy(H)

    loss = 0

    # Forward pass
    for t in range(len(X)):
        # gen char-vect
        x[t] = np.zeros((io_size, 1))
        x[t][X[t]] = 1

        # hidden_layer pass
        h[t] = np.tanh(Wxh @ x[t] + Whh @ h[t-1] + Bh)

        # y_layer pass (unnormalized)
        y[t] = Why @ h[t] + By

        # Normalized probs for next chars
        ey = np.exp(y[t])
        probs[t] = ey / np.sum(ey)

        # Softmax
        loss -= np.log(probs[t][Y[t], 0])
    
    # Deltas
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dBh  = np.zeros_like(Bh)
    dBy  = np.zeros_like(By)

    # Backpropagation
    for t in reversed(range(len(X))):

        # dC/dAy * dAy/dZy
        delta = np.copy(probs[t])
        delta[Y[t]] -= 1

        # dZy/dBy
        dBy += delta
        # dZy/dWhy
        dWhy += delta @ h[t].T

        # delta * dZy/dAh
        delta = Why.T @ delta
        # delta * dAh/dZh
        delta = (1 - h[t] * h[t]) * delta

        # dZh/dBh
        dBh += delta
        # dZh/dWxh
        dWxh += delta @ x[t].T 
        # dZh/dWhh
        dWhh += delta @ h[t-1].T
    
    # Clip to mitigate exploding gradients
    for d in [dWxh,dWhh,dWhy,dBh,dBy]:
        np.clip(d, -5, 5, out=d)
    
    return loss, dWxh, dWhh, dWhy, dBh, dBy, h[len(X)-1]

def sample(h, init_char_index, n):
    
    # gen char-vect
    x = np.zeros((io_size, 1))
    x[init_char_index] = 1

    char_indices = []

    # loop predictions
    for t in range(n):
        h = np.tanh(Wxh @ x + Whh @ h + Bh)
        y = Why @ h + By
        ey = np.exp(y)
        p = ey / np.sum(ey)

        next_char_index = np.random.choice(range(io_size), p=p.ravel())
        
        x = np.zeros((io_size, 1))
        x[next_char_index] = 1

        char_indices.append(next_char_index)
    
    txt = ''.join(int_to_char[i] for i in char_indices)
    print(f'{txt}\n-----')

# LET's TRAIN

n, p = 0, 0

# Mems
mWxh = np.zeros_like(Wxh)
mWhh = np.zeros_like(Whh)
mWhy = np.zeros_like(Why)
mBh  = np.zeros_like(Bh)
mBy  = np.zeros_like(By)

smooth_loss = -np.log(1.0/io_size) * sequence_length
state = None

while n <= 1e5:

    if p + sequence_length + 1 >= len(text) or n == 0:
        state = np.zeros((h_size, 1))
        p = 0
    
    X = [char_to_int[c] for c in text[p  :p+sequence_length  ]]
    Y = [char_to_int[c] for c in text[p+1:p+sequence_length+1]]

    loss, dWxh, dWhh, dWhy, dBh, dBy, state = backprop(X, Y, state)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if p == 0:
        print (f'-----\niter {n}, loss: {smooth_loss}')
        sample(state, X[0], 200)
    
    # parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, Bh, By],
                                [dWxh, dWhh, dWhy, dBh, dBy],
                                [mWxh, mWhh, mWhy, mBh, mBy]):
        mem += dparam * dparam
        param -= learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += sequence_length
    n += 1
