import tensorflow as tf
import numpy as np


# model to predict what the inputs*2 will be
# inputs = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
# targets = inputs*2

# model = tf.keras.Sequential([
# tf.keras.layers.Dense(4, activation='relu', input_shape=(1,)),
# tf.keras.layers.Dense(16),
# tf.keras.layers.Dense(16),
# tf.keras.layers.Dense(10),
# tf.keras.layers.Dense(1)
# ])

# model.compile(optimizer='adam', loss='mean_squared_error')

# model.fit(inputs, targets, epochs=5000)

# input_to_predict = np.array([120.0, 65.0, 12.0], dtype=float)
# predictions = model.predict(input_to_predict)
# print(predictions)


x = np.random.random((10000,1))*100-50
y = x**2

model = tf.keras.Sequential([
tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
tf.keras.layers.Dense(32),
tf.keras.layers.Dense(32),
tf.keras.layers.Dense(32),
tf.keras.layers.Dense(32),
tf.keras.layers.Dense(32),
tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x, y, epochs=15000, batch_size=256)

input_to_predict = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
predictions = model.predict(input_to_predict)
print(predictions)