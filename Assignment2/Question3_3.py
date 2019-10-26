import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

if __name__ == "__main__":

    X = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    Y = np.load("Y_train.npy")
    Y_test = np.load("Y_test.npy")

    svm_classifier = svm.SVC(kernel="linear")

    svm_classifier.fit(X, Y)
    predicted_X = svm_classifier.predict(X)
    predicted_X_test = svm_classifier.predict(X_test)

    empirical_loss = 0
    for i in range(len(X)):
        if predicted_X[i] != Y[i]:
            empirical_loss += 1

    empirical_loss = empirical_loss / len(X)

    true_loss = 0
    for i in range(len(X_test)):
        if predicted_X_test[i] != Y_test[i]:
            true_loss += 1

    true_loss = true_loss / len(X_test)

    print(empirical_loss)
    print(true_loss)
    print(svm_classifier.coef_)
