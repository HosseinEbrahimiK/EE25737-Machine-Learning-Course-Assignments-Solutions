import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


def violation(data_x, data_y, w):

    for k in range(len(data_x)):
        if data_y[k] * (np.matmul(w.T, data_x[k])) <= 0:
            return k

    return -1


if __name__ == "__main__":

    X = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    Y = np.load("Y_train.npy")
    Y_test = np.load("Y_test.npy")

    X = np.c_[np.ones(np.shape(X)[0]), X]
    X[:, 3] = X[:, 3] ** 3
    X_test = np.c_[np.ones(np.shape(X_test)[0]), X_test]
    X_test[:, 3] = X_test[:, 3] ** 3

    w = np.zeros(np.shape(X)[1])
    num_of_iteration = 10000
    loss_test = []
    iteration = []

    for i in range(1, num_of_iteration+1):

        ind = violation(X, Y, w)
        if ind != -1:
            w = w + Y[ind] * X[ind]
        else:
            break

        if i % 500 == 0:
            loss = 0
            for j in range(len(X_test)):
                if np.sign(np.matmul(w.T, X_test[j])) != Y_test[j]:
                    loss += 1
            loss_test.append(loss / len(X_test))
            iteration.append(i)

    final_loss = 0
    for i in range(len(X_test)):
        if np.sign(np.matmul(w.T, X_test[i])) != Y_test[i]:
            final_loss += 1

    final_loss = final_loss / len(X_test)

    print("list_of_loss = ", loss_test)
    print("final_loss = ", final_loss)
    print("final weights", w)

    plt.plot(iteration, loss_test)
    plt.ylabel("loss of test data")
    plt.xlabel("number of iterations")
    plt.show()
