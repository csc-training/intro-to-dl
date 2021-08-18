ex1_inputs = keras.Input(shape=(28, 28))

x = layers.Flatten()(ex1_inputs)
x = layers.Dense(units=50, activation="relu")(x)
x = layers.Dropout(rate=0.2)(x)
x = layers.Dense(units=50, activation="relu")(x)
x = layers.Dropout(rate=0.2)(x)

ex1_outputs = layers.Dense(units=10, activation='softmax')(x)
