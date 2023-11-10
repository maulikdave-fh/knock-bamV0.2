from flask import Flask, request
from collections import Counter
import soundfile as sf
import util
import tensorflow as tf
import numpy as np

app = Flask(__name__)

SAVED_MODEL_PATH_LOCAL = '../saved_models/brandisii_FCM.h5'
SAVED_MODEL_PATH_PROD = '/home/foresthut/mysite/saved_models/brandisii_FCM.h5'
SAMPLE_RATE = 44100
THRESHOLD = 98
MONO = 1


def _predict(samples):
    loaded_model = tf.keras.models.load_model(SAVED_MODEL_PATH_LOCAL)

    mfccs_scaled_features = util._extract_mfcc(samples)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    predicted_label = loaded_model.predict(mfccs_scaled_features)

    array = np.array(predicted_label) * 100

    classes_x = np.argmax(predicted_label, axis=1)

    if (np.max(array) < THRESHOLD):
        result = 'Unsure!'
    else:
        result = classes_x[0]

    print("Class-wise weightage - ", array)

    return str(result)


def _getPredictions(samples, sample_rate):
    segments = util._create_segmentsV1(samples)

    predictions = []

    print('Total {} segments detected..'.format(len(segments)))
    for i, segment in enumerate(segments):
        prediction = _predict(segment)

        print("Prediction for segment {} is {}".format(i+1, prediction))

        if prediction != 'Unsure!':
            predictions.append(prediction)

    return predictions


@app.route("/brandisii/v1.3", methods = ['POST'])
def brandisiiV1_3():
    fPCM = request.files['file']
    filename = fPCM.filename
    print('\n----------------------Request Received------------------------------------')

    samples, sample_rate = sf.read(fPCM, channels=MONO, samplerate=SAMPLE_RATE, format='RAW', subtype='PCM_16', endian='LITTLE')

    predictions = _getPredictions(samples, sample_rate)

    finalResult = 'Unsure!'

    if (len(predictions) > 1):
        counter = Counter(predictions)
        finalResult = counter.most_common()[0][0]
    elif len(predictions) == 1:
        finalResult = predictions[0]
    else:
        finalResult = 'Unsure!'

    if finalResult == '3':
        finalResult = 'Unsure!'

    print('Final Result: {}'.format(finalResult))

    return str(finalResult)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)