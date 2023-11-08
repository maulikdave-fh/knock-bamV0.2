import os
import librosa.feature
import soundfile as sf
from tqdm import tqdm
import json
import scipy
import numpy as np
import matplotlib.pyplot as plt

AUDIO_FILES_DIRECTORY = 'data/brandisii/raw/'
DATASET_PATH = 'data/brandisii/data.json'
SAMPLE_RATE = 44100
DURATION = 2  # in seconds
FRAME_SIZE = 1024
HOP_LENGTH = int(FRAME_SIZE / 4)
PRE_SET = 0.005  # start before onset - in seconds
NO_MFCC = 20


def _env_mask(wav, threshold=0.003):
    # Filter out noise
    # Absolute value
    wav = np.abs(wav)
    # Point wise mask determination.
    mask = wav > threshold
    return mask


def _create_segmentsV1(signal):
    segments = []

    peaks = scipy.signal.find_peaks(signal, height=0.1, distance=1000)
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


def _extract_zcr(segment):
    return np.std(librosa.feature.zero_crossing_rate(y=segment, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH))


def _extract_rms(segment):
    S, phase = librosa.magphase(librosa.stft(segment, hop_length=HOP_LENGTH, n_fft=FRAME_SIZE))
    rms = librosa.feature.rms(S=S, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)
    return np.std(rms)


def _extract_spec_rolloff_lower(segment):
    return np.std(librosa.feature.spectral_rolloff(y=segment, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                n_fft=FRAME_SIZE, roll_percent=0.15))


def _extract_spec_rolloff_higher(segment):
    return np.std(librosa.feature.spectral_rolloff(y=segment, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                n_fft=FRAME_SIZE, roll_percent=0.85))


def _extract_spec_flatness(segment):
    return np.std(librosa.feature.spectral_flatness(y=segment, hop_length=HOP_LENGTH, n_fft=FRAME_SIZE))


def _extract_spec_centroid(segment):
    return np.std(librosa.feature.spectral_centroid(y=segment, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                             n_fft=FRAME_SIZE))


def _extract_mfcc(segment):
    mfccs = librosa.feature.mfcc(y=segment, hop_length=HOP_LENGTH, n_fft=FRAME_SIZE, n_mfcc=NO_MFCC)
    mfccs = mfccs.T
    mfccs_std = np.std(mfccs, axis=0)
    return mfccs_std


def _create_dataset():
    # Define data structure
    data = {
        'labels': [],
        'zcr': [],
        'rms': [],
        'spec_rolloff_lower' : [],
        'spec_rolloff_higher' : [],
        'spec_flatness' : [],
        'spec_centroid' : [],
        'mfcc' : []
    }

    # Iterate through files
    for dirname, _, filenames in os.walk(AUDIO_FILES_DIRECTORY):
        for filename in tqdm(filenames, 'Extracting Features from audio files in directory {}....'.format(dirname)):
            filepath = os.path.join(dirname, filename)
            signal, sr = sf.read(filepath)

            # Extract label from filename
            label = filepath[filepath.rfind('/') + 1: filepath.rfind('_')]

            # Split audio file into smaller chunks
            segments = _create_segmentsV1(signal)

            # Iterate over segments to extract features
            for i, segment in enumerate(segments):
                # Add Label
                data['labels'].append(int(label))

                # Add zcr
                data['zcr'].append(_extract_zcr(segment))

                # Add rms
                data['rms'].append(_extract_rms(segment))

                # Add Spectral Rolloff lower
                data['spec_rolloff_lower'].append(_extract_spec_rolloff_lower(segment))

                # Add Spectral Rolloff higher
                data['spec_rolloff_higher'].append(_extract_spec_rolloff_higher(segment))

                # Add Spectral Flatness
                data['spec_flatness'].append(_extract_spec_flatness(segment))

                # Add Spectral Centroid
                data['spec_centroid'].append(_extract_spec_centroid(segment))

                # Add MFCC
                data['mfcc'].append(_extract_mfcc(segment).tolist())

    # Save to JSON
    with open(DATASET_PATH, 'w') as fp:
        json.dump(data, fp, indent=4)


def _plt_dataset():
    with open(DATASET_PATH, 'r') as dataset_file:
        data = json.load(dataset_file)

    X = data['mfcc']
    y = data['labels']

    labels, counts = np.unique(y, return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.suptitle('Class-wise Data')
    plt.show()

if __name__ == '__main__':
    _create_dataset()
    _plt_dataset()