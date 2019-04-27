import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Training data
x_train, y_train = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.optimizers import Adam

I_layer = Input((x_train.shape[1],))
H_layer = Dense(10, activation='relu')(I_layer)
H_layer = Dense(10, activation='relu')(H_layer)
O_layer = Dense(1, activation='sigmoid')(H_layer)
model = Model(inputs=I_layer, outputs=O_layer)
model.compile(Adam(lr=0.09), 'binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train)


# Plot
res = 80
_x = np.linspace(-1.5, 1.5, res)
_y = np.linspace(-1.5, 1.5, res)
_Y = np.zeros((res,res))

for i, _x_ in enumerate(_x):
    for j, _y_ in enumerate(_y):
            _Y[i, j] = model.predict(np.array([[_x_,_y_]]))

# Display plot
plt.pcolormesh(_x, _y, _Y, cmap='coolwarm')
plt.axis('equal')
plt.scatter(x_train[y_train[:] == 0, 0], x_train[y_train[:] == 0, 1], c='skyblue')
plt.scatter(x_train[y_train[:] == 1, 0], x_train[y_train[:] == 1, 1], c='salmon')
plt.show()