from __future__ import print_function
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import os.path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Embedding, Activation, Lambda, Bidirectional
from keras.layers import LSTM, GRU
from keras.optimizers import SGD
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from IPython import embed
import dataset
from vars_local import *

batch_size = 10000 
epochs = 5
LEARNING_RATE = 1e-4
NUM_CLASSES = None  # it will be set dynamically

def load_data():
    ds = pd.read_csv(TRAIN_PATH)
    ds = dataset.prepare_features(ds)
    
    dataset_y = ds.pop("Cancer_Type")
    dataset_x = ds 

    # Encode labels before splitting
    encoder = LabelEncoder()
    dataset_y = encoder.fit_transform(dataset_y)

    # Determine correct number of classes
    global NUM_CLASSES
    NUM_CLASSES = len(set(dataset_y))  # Ensure NUM_CLASSES matches actual data

    print("Unique class labels after encoding:", set(dataset_y))
    print("NUM_CLASSES:", NUM_CLASSES)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=0)

    return (X_train.to_numpy(), y_train, X_test.to_numpy(), y_test)

(train_x, train_y, test_x, test_y) = load_data()

# Convert labels to categorical
train_y = keras.utils.to_categorical(train_y, NUM_CLASSES)
test_y = keras.utils.to_categorical(test_y, NUM_CLASSES)

shape = train_x.shape

model = Sequential()
model.add(Dense(46, input_shape=(shape[1],), activation='selu'))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y,
          batch_size=batch_size,
          shuffle=True,
          epochs=20,
          verbose=1,
          validation_data=(test_x, test_y))

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
embed()