import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
np.set_printoptions(threshold=np.inf)


if __name__ == "__main__":

    trainData = []
    with open('data/Train/X_train.txt') as input_file:
        for line in input_file:
            trainData.append(list(map(float, line.strip().split(',')[0].split())))
    input_file.close()

    testData = []
    with open('data/Train/y_train.txt') as input_file:
        for line in input_file:
            testData.append(float(line.strip().split(',')[0]))
    input_file.close()

    X = np.array(trainData[:int(0.9 * len(trainData))])
    Y = np.array(testData[:int(0.9 * len(testData))])

    X_test = np.array(trainData[int(0.9 * len(trainData)):])
    Y_test = np.array(testData[int(0.9 * len(trainData)):])

    LossS = []
    LossT = []
    n = [1, 2, 4, 8, 16, 32]
    for i in n:

        mlp = MLPClassifier(hidden_layer_sizes=tuple(i for _ in range(8)), max_iter=100)
        mlp.fit(X, Y)
        prediction_s = mlp.predict(X)
        loss_s = 0
        for j in range(len(Y)):
            if prediction_s[j] != Y[j]:
                loss_s += 1

        LossS.append(loss_s / len(Y))

        prediction_t = mlp.predict(X_test)
        loss_t = 0
        for j in range(len(Y_test)):
            if prediction_t[j] != Y_test[j]:
                loss_t += 1

        LossT.append(loss_t / len(Y_test))

    plt.scatter(n, LossS, color="b")
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(n, LossS), key=sort_axis)
    x, y = zip(*sorted_zip)
    plt.plot(x, y, color='b')

    plt.scatter(n, LossT, color="r")
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(n, LossT), key=sort_axis)
    x, y = zip(*sorted_zip)
    plt.plot(x, y, color='r')
    plt.legend(["Empirical Risk", "True Risk"])
    plt.xlabel("Number of nodes in each layer")
    plt.ylabel("Loss")
    plt.show()
    print(LossS)
    print(LossT)