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


def loadFeatures(files):
    extractor = FeaturesExtractor.Extractor()
    features = np.asarray(())
    for file in files:
        feature = extractor.extractFeatures(file)
        if features.size == 0:
            features = feature
        else:
            features = np.vstack((features, feature))
    return features


def save(destination, model):
    pickle.dump(model, open(destination + ".gmm", 'wb'))
    print("Model Saved")


def trainModel(features, destination):
    model = GaussianMixture(n_components=8, max_iter=200, covariance_type='diag', n_init=3)
    model.fit(features)
    save(destination, model)


def main():
    labels = ['males', 'females']
    source = "data/"
    destination = "data/saved/"
    for label in labels:
        actualSource = source + label + "/"
        actualDestination = destination + label
        print("Loading {} files ".format(label))
        files = [os.path.join(actualSource, f) for f in os.listdir(actualSource) if f.endswith('.wav')]
        features = loadFeatures(files)
        trainModel(features, actualDestination)
        print("Done {} model".format(label))


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
