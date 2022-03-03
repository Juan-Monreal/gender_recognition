import numpy as np
import pandas as pd
import os
import pickle
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
import python_speech_features as mfcc
from python_speech_features import delta
from sklearn import preprocessing
import FeaturesExtractor
import warnings

warnings.filterwarnings("ignore")


# ignore WARNING:root:frame length (2400) is greater than FFT size (512), frame will be truncated. Increase NFFT to avoid.

# def get_features(rate, audio):
#     """
#     Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
#     :param rate: audio signal
#     :param audio: the samplerate of the signal
#     :return: Extracted features
#     """
#     # https://python-speech-features.readthedocs.io/en/latest/index.html#functions-provided-in-python-speech-features-module
#     features = mfcc.mfcc(audio, rate, winlen=0.025, winstep=0.01, numcep=13, appendEnergy=False)
#     features = preprocessing.scale(features)
#     return features

extractor = FeaturesExtractor.Extractor()

source = "data/females/"
# source = "data/males/"
destination = "data/saved/"

files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.wav')]

features = np.asarray(())
print("Loading Files")
for file in files:
    feature = extractor.extractFeatures(file)
    if features.size == 0:
        features = feature
    else:
        features = np.vstack((features, feature))
model = GaussianMixture(n_components=8, max_iter=200, covariance_type='diag', n_init=3)
model.fit(features)
# pickleFile = "male.gmm"
pickleFile = "female.gmm"

pickle.dump(model, open(destination + pickleFile, 'wb'))
print("done")
