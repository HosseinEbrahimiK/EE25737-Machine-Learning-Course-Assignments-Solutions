import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    df = pd.read_csv('fashion-mnist.csv')

    Y = np.array(df.loc[:, 'y'].copy())
    X = np.array(df.drop(['y'], axis=1).copy())

    idx = np.arange(10000)
    idx_train = np.random.choice(10000, size=5000, replace=False)
    idx_valid = np.delete(idx, idx_train)

    X_train = X[idx_train, :]
    Y_train = Y[idx_train]

    X_valid = X[idx_valid, :]
    Y_valid = Y[idx_valid]

# C5.1

    linearSVM = svm.LinearSVC()
    linearSVM.fit(X_train, Y_train)
    linearSVM_valid_predict = linearSVM.predict(X_valid)
    cm_valid = confusion_matrix(Y_valid, linearSVM_valid_predict)
    accuracy = (np.sum(np.diagonal(cm_valid)) / np.sum(cm_valid)) * 100
    print(cm_valid)
    print(accuracy)

# C5.2

    gaussianSVM = svm.SVC(gamma=38*1e-8, kernel='rbf')
    gaussianSVM.fit(X_train, Y_train)
    prediction = gaussianSVM.predict(X_valid)
    cm_valid = confusion_matrix(Y_valid, prediction)
    accuracy = (np.sum(np.diagonal(cm_valid)) / np.sum(cm_valid)) * 100
    print(accuracy)
    print(cm_valid)

# C5.3

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, Y_train)
    prediction = knn.predict(X_valid)
    cm_valid = confusion_matrix(Y_valid, prediction)
    accuracy = (np.sum(np.diagonal(cm_valid)) / np.sum(cm_valid)) * 100
    print(accuracy)
    print(cm_valid)

# C5.4

    DTC = DecisionTreeClassifier()
    DTC.fit(X_train, Y_train)
    prediction = DTC.predict(X_valid)
    cm_valid = confusion_matrix(Y_valid, prediction)
    accuracy = (np.sum(np.diagonal(cm_valid)) / np.sum(cm_valid)) * 100
    print(accuracy)
    print(cm_valid)

# C5.5

    model = Sequential()
    n_cols = X_train.shape[1]
    model.add(Dense(100, activation='sigmoid', input_dim=n_cols))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(X_train, Y_train, epochs=100)
    matrix = model.predict(X_valid)
    prediction = np.argmax(matrix, axis=1)
    cm_valid = confusion_matrix(Y_valid, prediction)
    accuracy = (np.sum(prediction == Y_valid) / 5000) * 100
    print(accuracy)
    print(cm_valid)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
