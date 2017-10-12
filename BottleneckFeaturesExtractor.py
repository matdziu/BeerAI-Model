import json
import math

import numpy as np
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator

PROJECT_PATH = "/Users/mateuszdziubek/Desktop/BeerAI-Model"

datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 16
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
validation_generator = datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

bottleneck_features_extractor = InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

output_file = open(f"{PROJECT_PATH}/models/bottleneck_features_extractor.txt", 'w')
json.dump(bottleneck_features_extractor.to_json(), output_file)

bottleneck_features_train = bottleneck_features_extractor.predict_generator(train_generator,
                                                                            math.ceil(
                                                                                train_generator.classes.size / batch_size))
bottleneck_features_validation = bottleneck_features_extractor.predict_generator(validation_generator,
                                                                                 math.ceil(
                                                                                     validation_generator.classes.size / batch_size))

np.save('bottleneck_features_train.npy', bottleneck_features_train)
np.save('bottleneck_labels_train.npy', np.array(train_generator.classes))
np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
np.save('bottleneck_labels_validation.npy', np.array(validation_generator.classes))
