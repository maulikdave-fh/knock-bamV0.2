import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

DATASET_PATH = 'data/brandisii/data.json'


def _load_data(path):
    """Loads training dataset from json file.

        :param path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    with open(path, 'r') as dataset_file:
        data = json.load(dataset_file)

    X = np.array(data['mfcc'])
    y = np.array(data['labels'])

    print('Data loaded!')

    return X, y


if __name__ == '__main__':
    # load data
    X, y = _load_data(DATASET_PATH)

    print(X.shape)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # build network
    model = tf.keras.Sequential([

        # input layer
        tf.keras.layers.Flatten(input_shape=(X.shape[1], 1)),

        # 1st dense layer
        tf.keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        tf.keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        tf.keras.layers.Dense(64, activation='relu'),

        # output layer
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    for train, test in kfold.split(X_train, y_train):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=100)

    # evaluate the model
    print('\n-----------------Model Evaluation---------------')
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Test loss: ", scores[0])
    print("Test accuracy: ", scores[1] * 100)