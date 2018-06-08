import numpy as np
import keras
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_in = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

logic_and = np.array([[0],
                      [0],
                      [0],
                      [1]])

logic_or = np.array([[0],
                     [1],
                     [1],
                     [1]])

logic_xor = np.array([[0],
                      [1],
                      [1],
                      [0]])

model = keras.models.Sequential(layers=[
    keras.layers.Dense(input_dim=2, units=2),
    keras.layers.Activation(keras.activations.sigmoid),
    keras.layers.Dense(units=1),
    keras.layers.Activation(keras.activations.sigmoid)
])

model.compile(optimizer=keras.optimizers.SGD(lr=.5), loss='mse')

model.fit(data_in, logic_and)


