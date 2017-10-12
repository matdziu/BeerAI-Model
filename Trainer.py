import json

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical

PROJECT_PATH = "/Users/mateuszdziubek/Desktop/BeerAI-Model"

num_classes = 10
batch_size = 48
epochs = 50

train_data = np.load('bottleneck_features_train.npy')
train_labels = to_categorical(np.load('bottleneck_labels_train.npy'), num_classes)
validation_data = np.load('bottleneck_features_validation.npy')
validation_labels = to_categorical(np.load('bottleneck_labels_validation.npy'), num_classes)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

output_file = open(f"{PROJECT_PATH}/models/model.json", 'w')
json.dump(model.to_json(), output_file)

model_checkpoint = ModelCheckpoint(filepath='beer_label_classifier_weights.h5', verbose=1, save_best_only=True)
model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels),
          callbacks=[model_checkpoint])
