import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

if __name__ == "__main__":
    train = pd.read_csv(r"train.csv", lineterminator="\n")
    valid = pd.read_csv(r"validation.csv", lineterminator="\n")

    x_train = train['X_train'].copy()
    y_train = train['Y_train\r'].copy()

    x_valid = valid['X_validation'].copy()
    y_valid = valid['Y_validation\r'].copy()

    mean_x_train = sum(x_train) / len(x_train)
    variance = 0
    for i in range(len(x_train)):
        variance += (x_train[i] - mean_x_train) ** 2

    variance /= (len(x_train) - 1)

    std = math.sqrt(variance)

    for i in range(len(x_train)):
        x_train[i] = (x_train[i] - mean_x_train) / std

    for i in range(len(x_valid)):
        x_valid[i] = (x_valid[i] - mean_x_train) / std

    X = np.array(x_train.values)
    Y = np.array(y_train.values)

    X_valid = np.array(x_valid)
    Y_valid = np.array(y_valid)

    weights = list()
    empirical_risk = list()
    true_risk = list()

    for i in range(1, 16):

        arr = [[0 for _ in range(i+1)] for _ in range(len(X))]
        arr_valid = [[0 for _ in range(i+1)] for _ in range(len(X_valid))]

        for j in range(len(X)):
            for k in range(i+1):
                arr[j][k] = X[j] ** k

        for j in range(len(X_valid)):
            for k in range(i+1):
                arr_valid[j][k] = X_valid[j] ** k

        Psi = np.array(arr)
        Psi_valid = np.array(arr_valid)

        A = np.matmul(Psi.T, Psi)
        b = np.matmul(Psi.T, Y)
        weights.append(np.matmul(np.linalg.inv(A), b))

        y_show = []
        loss = 0

        for j in range(len(X)):
            loss += (np.matmul(weights[i-1].T, Psi[j]) - Y[j]) ** 2
            y_show += [np.matmul(weights[i-1].T, Psi[j])]

        empirical_risk.append(loss / len(X))

        loss_valid = 0

        for j in range(len(X_valid)):
            loss_valid += (np.matmul(weights[i-1].T, Psi_valid[j]) - Y_valid[j]) ** 2

        true_risk.append(loss_valid / len(X_valid))

        plt.scatter(X, Y, color="b")
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(X, y_show), key=sort_axis)
        x, y_show = zip(*sorted_zip)
        plt.plot(x, y_show, color='r')
        plt.show()

    plt.figure(1)
    plt.subplot(211)
    plt.plot([i for i in range(1, 16)], empirical_risk)

    plt.subplot(212)
    plt.plot([i for i in range(1, 16)], true_risk)
    plt.show()
    print("weights", weights)
    print("empirical_risk = ", empirical_risk)
    print("true_risk = ", true_risk)
