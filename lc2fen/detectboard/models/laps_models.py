"""This module works with the LAPS models."""


from keras.models import model_from_json


__laps_model = "laps.model.json"
__laps_weights = "laps.weights.h5"
LAPS_MODEL = model_from_json(open(__laps_model, "r").read())
LAPS_MODEL.load_weights(__laps_weights)
LAPS_MODEL.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
LAPS_MODEL.save("laps_model.h5")

"""
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import *

# input
model = Sequential()
model.add(Dense(441, input_shape=(21,21,1)))

# H(2)
for i in range(2):
    for j in [3, 2, 1]:
        model.add(Conv2D(16, j, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

# F(128)
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())

# output
model.add(Dense(2, activation='softmax'))
model.compile(RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
"""
