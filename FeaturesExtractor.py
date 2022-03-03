import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
from python_speech_features import mfcc
from python_speech_features import delta


class Extractor:

    def __init__(self):
        pass

    def extractFeatures(self, path: str):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        :param path: path to wave file
        :return: Extracted features matrix
        """
        rate, audio = read(path)
        # mfcc doc
        # https://python-speech-features.readthedocs.io/en/latest/index.html#functions-provided-in-python-speech-features-module
        mfccFeatures = mfcc(
            audio,
            rate,
            winlen=0.05,
            winstep=0.01,
            numcep=5,
            nfilt=30,
            nfft=512,
            appendEnergy=True)
        mfccFeatures = preprocessing.scale(mfccFeatures)  # Compute delta features from a feature vector sequence.
        deltas = delta(mfccFeatures, 2)  # Compute delta features from a feature vector sequence.
        double_deltas = delta(deltas, 2)
        combined = np.hstack((mfccFeatures, deltas, double_deltas))
        return combined
