import numpy as np
from numpy import savetxt
import os
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from FeaturesExtractor import Extractor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")  # It is supposed to ignore all warnings, but sometimes it doesn't work.


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
    Check if a given list is empty or not
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
    """
    Initialize the models with Training data using trianModel method
    :param labels: list of string containing the genders
    :param source: path to obtain the files
    :param destination: path to save data
    """
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
    """
    Get the trained Models for males and females.
    Then obtain all the training data (.wav files).
    Next we required the log likelihood scores for each one.
    And select the best according to the trained model
    :param trainedModels: list of paths with the current trained models (.gmm files)
    :param source: path to obtain the files
    :param labels: list of string containing the genders
    """
    models = [pickle.load(open(file, 'rb')) for file in trainedModels]
    genders = [file.split("\\")[-1].split(".gmm")[0] for file in trainedModels]
    extractor = Extractor()
    print(genders)
    winners = []  # prediction
    n = []
    for label in labels:
        actualSource = source + label + "/"
        print(actualSource, 'Actual source')
        files = getAllFiles('.wav', actualSource)
        n.append(len(files))
        for file in files:
            print(file.split("\\")[-1])
            features = extractor.extractFeatures(file)
            likeLihood = np.zeros(len(models))
            for x in range(len(models)):
                model = models[x]
                scores = np.array(model.score(features))
                likeLihood[x] = scores.sum()
            winner = np.argmax(likeLihood)
            # genders[winner].split('/')[2]
            winners.append(genders[winner].split('/')[2])
            print("Ganador: ", genders[winner].split('/')[2], "Mujeres: ", likeLihood[0], "Hombres: ", likeLihood[1],
                  "\n")
    targets = ['males'] * n[0]
    targets.extend(['females'] * n[1])
    drawConfusionMatrix(targets, winners)


def drawConfusionMatrix(test, prediction):
    """
    Draw a confusion matrix using sklearn.metrics package.
    The confusion_matrix() method will give you an array that depicts the
    True Positives, False Positives, False Negatives, and True negatives.
    Once we have the confusion matrix created, we use the heatmap() method available in the
    seaborn library to plot the confusion matrix
    :param test: list of label's correctly labeled
    :param prediction: list of the label's predicted
    """
    print("TEST", len(test))
    print(test)
    print("prediction", len(prediction))
    print(prediction)
    confusion = confusion_matrix(test, prediction)
    ax = sns.heatmap(confusion, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(['females', 'males'])
    ax.yaxis.set_ticklabels(['females', 'males'])
    ax.figure.savefig("data/saves/confusionMatrix.png", dpi=300)
    plt.show()


def main():
    """
    Main Function
    Initialize the models if not are previously generated
    Then goes to test the models with testing data
    """
    labels = ['males', 'females']
    source = "data/trainingData/"
    destination = "data/saves/"
    trainedModels = getAllFiles('.gmm', destination)
    if listIsEmpty(trainedModels):
        initializeModel(labels, source, destination)
        trainedModels = getAllFiles('.gmm', destination)
    testModel(trainedModels, "data/testingData/", labels)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
