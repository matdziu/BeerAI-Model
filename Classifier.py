import numpy as np
from keras.applications import InceptionV3
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array

output_beer_labels = ['Harnaś', 'Kasztelan Niepasteryzowany', 'Kasztelan Pszeniczny', 'Miłosław Witbier (niebieski)',
                      'Perła Chmielowa', 'Perła Export', 'Somersby', 'Warka', 'Wojak', 'Żywiec Biały']

image_path = "test.jpg"
image = load_img(image_path, target_size=(150, 150))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = image / 255

model = InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
bottleneck_feature = model.predict(image)

model = Sequential()
model.add(Flatten(input_shape=bottleneck_feature.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output_beer_labels), activation='softmax'))
model.load_weights('beer_label_classifier_weights.h5')

prediction_encoded = model.predict(bottleneck_feature)
print(f"Prediction: {output_beer_labels[np.argmax(prediction_encoded)]}")
