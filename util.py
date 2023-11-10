import numpy as np
import scipy
import librosa
import json

SAMPLE_RATE = 44100
DURATION = 2  # in seconds
FRAME_SIZE = 1024
HOP_LENGTH = int(FRAME_SIZE / 4)
PRE_SET = 0.005  # start before onset - in seconds
NO_MFCC = 20
DATASET_PATH = 'data/brandisii/data.json'

def _load_dataset(path=DATASET_PATH):
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


def _env_mask(signal, threshold=0.003):
    """Returns mask to filter noise from given signal.

        :param signal (ndarray): Input Raw Signal
        :param threshold (float): Threshold below which the sample to considered noisy
        :returns mask (ndarray): Mask to be applied to the original signal to remove unwanted samples

    """
    # Absolute value
    signal = np.abs(signal)
    # Point wise mask determination.
    mask = signal > threshold
    return mask

def _create_segmentsV1(signal):
    segments = []

    peaks = scipy.signal.find_peaks(signal, height=0.3, distance=1000)
    peaks_list = peaks[1]['peak_heights'].tolist()

    for i, peak in enumerate(peaks_list):
        sample_no_at_peak = peaks[0][i]
        start = int(sample_no_at_peak - (PRE_SET * SAMPLE_RATE))
        end = int(sample_no_at_peak + ((DURATION - PRE_SET) * SAMPLE_RATE))

        segment = signal[start: end]
        segment_masked = segment[_env_mask(segment)]

        if len(segment_masked) >= FRAME_SIZE:
            segments.append(segment_masked)
        # print('\nSegment length before masking {} and after masking {}'.format(len(segment), len(segment_masked)))
    return segments

def _extract_mfcc(segment):
    mfccs = librosa.feature.mfcc(y=segment, hop_length=HOP_LENGTH, n_fft=FRAME_SIZE, n_mfcc=NO_MFCC)
    mfccs = mfccs.T
    mfccs_std = np.std(mfccs, axis=0)
    return mfccs_std