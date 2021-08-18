ex1_inputs = keras.Input(shape=input_shape)

x = layers.Conv2D(32, (3, 3),
                  padding='valid',
                  activation ='relu')(ex1_inputs)
x = layers.Conv2D(32, (3, 3),
                  padding='valid',
                  activation ='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(units=128, activation ='relu')(x)
x = layers.Dropout(0.5)(x)

ex1_outputs = layers.Dense(units=nb_classes,
                       activation='softmax')(x)