import json
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

LEARNING_RATE = 0.0001
N_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 32
DATASET_PATH = 'data/brandisii/data.json'
LABELS = ['New', 'One', 'Two', 'Noise']

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


def _plt_metrics(predictions, y_true, history):
    fig, axs = plt.subplots(3, 1, tight_layout=True)
    fig.suptitle('Model Metrics')

    # Confusion Matrix
    y_pred = tf.argmax(predictions, axis=1).numpy()
    axs[0].set_title('Confusion Matrix')
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1, 2, 3], display_labels=LABELS, ax=axs[0])

    # Loss Entropy
    metrics = history.history
    axs[1].plot(history.epoch, metrics['loss'], metrics['val_loss'])
    axs[1].legend(['loss', 'val_loss'])
    axs[1].set_ylim([0, max(plt.ylim())])
    axs[1].set_title('Loss Entropy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss [CrossEntropy]')

    # Accuracy Entropy
    axs[2].plot(history.epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
    axs[2].legend(['accuracy', 'val_accuracy'])
    axs[2].set_ylim([0, 100])
    axs[2].set_title('Accuracy Entropy')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Accuracy [%]')

    plt.show()


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
    optimiser = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

    # introduce checkpoints
    checkpointer_cb = ModelCheckpoint(filepath='saved_models/brandisii_FCM.h5',
                                      verbose=1, save_best_only=True, save_weights_only=False, monitor="val_loss")
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', verbose=1)
    earlystopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

    histories = []
    for train, test in kfold.split(X_train, y_train):
        history = model.fit(X_train,
                            y_train,
                            validation_data=(X_val, y_val),
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            callbacks=[checkpointer_cb, reduce_lr_cb, earlystopping_cb]
                            )
        histories.append(history)

    # evaluate the model
    print('\n-----------------Model Evaluation---------------')
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Test loss: ", scores[0])
    print("Test accuracy: ", scores[1] * 100)

    # plot metrics
    print('\n----------------Model Metrics----------------')
    # Passing the first history instance, from the first iteration to see how loss & accuracy shape up.
    # If passed the last one, it's all straight horizontal line :-)
    _plt_metrics(model.predict(X_test), y_test, histories[0])