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
import _pickle as cPickle


def loadFeatures(files):
    """
    Go iterative for each file(.wav) in loaded files (*.wav) and
    extract the MFCC Features using the Extractor() class in FeaturesExtractor
    :param files: List of each File to be processed
    :return: All features for each audio in a numpy matrix
    """
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
    """
    Serializes the trained model using the Pickle Library
    The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
    “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream,
    and “unpickling” is the inverse operation
    :param destination: path to save the model
    :param model:  trained model to be saved
    """
    pickle.dump(model, open(destination + ".gmm", 'wb'))
    print("Model Saved")


def trainModel(features, destination):
    """
    Train the current gender model using Gaussian Mixture.
    Gaussian Mixture will try to learn their distribution, which will be representative
    of the gender.
    :param features: mfcc features for each audio
    :param destination: path to save the model
    """
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
        a = trainModel(features, actualDestination)
        print("Done {} model".format(label))


    gmm_files = [os.path.join(destination,fname) for fname in 
              os.listdir(destination) if fname.endswith('.gmm')]

    models = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
    genders   = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
    # print(gmm_files)
    # print(models)
    for f in files:
         print(f.split("\\")[-1])
         sr, audio =read(f)
         scores = None
         log = np.zeros(len(models))
         for x in range(len(models)):
            gmm = models[x]
            scores = np.array(gmm.score(features))
            log[x] = scores.sum()

    winner = np.argmax(log)
    print("Ganador: ", genders[winner], "Mujeres: ", log[0], "Hombres: ", log[1],"\n" )


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
