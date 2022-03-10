import numpy as np
from numpy import savetxt
import os
import pickle
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from FeaturesExtractor import Extractor
import warnings
warnings.filterwarnings("ignore")


def loadFeatures(files):
    """
    Go iterative for each file(.wav) in loaded files (*.wav) and
    extract the MFCC Features using the Extractor() class in FeaturesExtractor
    :param files: List of each File to be processed
    :return: All features for each audio in a numpy matrix
    """
    extractor = Extractor()
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


def listIsEmpty(files) -> bool:
    """
    Chec if a given list is empty or not
    :param files: list to be checked
    :return: True if list is empty
            Else If is not empty
    """
    if files is None:
        return True
    if not files:
        return True
    return False


def getAllFiles(extension, path):
    """
    Get all the files with the same extension
    :param extension: extension file to be obtained
    :param path: directory/path to check
    :return: List with the path of files
    """
    return [os.path.join(path, file)
            for file in os.listdir(path)
            if file.endswith(extension)]


def initializeModel(labels, source, destination):
    for label in labels:
        actualSource = source + label + "/"
        actualDestination = destination + label
        print("Loading {} files ".format(label))
        files = getAllFiles('.wav', actualSource)
        features = loadFeatures(files)
        savetxt(destination + label + '_data.csv', features, delimiter=',')
        trainModel(features, actualDestination)
        print("{} model done".format(label))


def testModel(trainedModels, source, labels):
    models = [pickle.load(open(file, 'rb')) for file in trainedModels]
    genders = [file.split("\\")[-1].split(".gmm")[0] for file in trainedModels]
    files = getAllFiles('.wav', source + 'males')
    extractor = Extractor()
    print(genders)
    winners = list()
    for file in files:
        print(file.split("\\")[-1])
        features = extractor.extractFeatures(file)
        likeLihood = np.zeros(len(models))
        for x in range(len(models)):
            model = models[x]
            scores = np.array(model.score(features))
            likeLihood[x] = scores.sum()

        winner = np.argmax(likeLihood)
        #genders[winner].split('/')[2]
        winners.append(genders[winner].split('/')[2])
        # print("Ganador: ", genders[winner].split('/')[2], "Mujeres: ", likeLihood[0], "Hombres: ", likeLihood[1], "\n")


def main():
    labels = ['males', 'females']
    source = "data/trainingData/"
    destination = "data/saves/"
    trainedModels = getAllFiles('.gmm', destination)
    if listIsEmpty(trainedModels):
        print("if")
        initializeModel(labels, source, destination)
        trainedModels = getAllFiles('.gmm', destination)
    testModel(trainedModels, source, labels)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
