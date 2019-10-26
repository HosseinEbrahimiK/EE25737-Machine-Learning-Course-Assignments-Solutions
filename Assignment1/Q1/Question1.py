import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


if __name__ == "__main__":
    dataFrame = pd.read_csv(r"data_Q1.csv", lineterminator="\n")

    data_y = dataFrame['medianHouseValue\r'].copy()
    data_x = dataFrame.drop(columns=['medianHouseValue\r']).copy()

    name_cols = data_x.columns.values.tolist()

    data_feature = np.array(data_x.values)
    data_label = np.array(data_y.values)

    X, X_test = data_feature[:int(0.75 * len(data_feature))], data_feature[int(0.75 * len(data_feature)):]
    Y, Y_test = data_label[:int(0.75 * len(data_label))], data_label[int(0.75 * len(data_label)):]

    for j in range(8):
        plt.plot(data_feature[:, j], data_label, 'b.')
        plt.xlabel(name_cols[j])
        plt.ylabel('medianHouseValue')
        plt.show()

    X = np.c_[np.ones(np.shape(X)[0]), X]
    X_test = np.c_[np.ones(np.shape(X_test)[0]), X_test]

    A = np.matmul(X.T, X)
    b = np.matmul(X.T, Y)
    w = np.matmul(np.linalg.inv(A), b)
    print("weights = ", w)

    empirical_risk = 0
    true_risk = 0

    for i in range(len(X)):
        loss = np.matmul(w.T, X[i]) - Y[i]
        empirical_risk += (loss * loss)

    empirical_risk = empirical_risk / len(X)

    for i in range(len(X_test)):
        loss = np.matmul(w.T, X_test[i]) - Y_test[i]
        true_risk += (loss * loss)

    true_risk = true_risk / len(X_test)

    print("empirical_risk = ", empirical_risk)
    print("true_risk = ", true_risk)
