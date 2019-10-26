import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------- 6.a ----------------------------------
def k_means(matrix, k, max_num_iterations=100):

    np.random.RandomState(123)
    random_idx = np.random.permutation(matrix.shape[0])
    centroid = matrix[random_idx[:k]]

    hist = list()
    hist.append(centroid.copy())

    clusters = 0
    for _ in range(max_num_iterations):
        clusters = [[] for _ in range(k)]
        for v in matrix:
            min_dist = np.inf
            index = 0
            for i in range(k):
                distance = np.linalg.norm(v - centroid[i])
                if distance < min_dist:
                    min_dist = distance
                    index = i

            clusters[index].append(v)

        for i in range(k):
            if len(clusters[i]) != 0:
                centroid[i] = np.sum(clusters[i], axis=0) / len(clusters[i])

        hist.append(centroid.copy())

    return hist, clusters


if __name__ == "__main__":

    # -------------- 6.b ------------------------------
    df = pd.read_csv('iris.csv')

    X = np.array(df.drop(['variety'], axis=1).copy())
    hist_centroids, classes = k_means(X, 3)

    features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

    for i in range(len(classes)):
        classes[i] = np.array(classes[i])

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            plt.scatter(classes[0][:, i], classes[0][:, j], color='red')
            plt.scatter(classes[1][:, i], classes[1][:, j], color='blue')
            plt.scatter(classes[2][:, i], classes[2][:, j], color='green')
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.show()

    # -------------- 6.d -----------------------------
    X_prime = X[:, [2, 3]]
    centrals, b_class = k_means(X_prime, 3)
    centroids = [[] for _ in range(3)]

    for i in range(len(centrals)):
        for j in range(len(centrals[i])):
            centroids[j].append(centrals[i][j])

    for i in range(len(centroids)):
        centroids[i] = np.array(centroids[i])

    plt.scatter(centroids[0][:, 0], centroids[0][:, 1], color='red')
    plt.scatter(centroids[1][:, 0], centroids[1][:, 1], color='blue')
    plt.scatter(centroids[2][:, 0], centroids[2][:, 1], color='green')

    print("centroid No. 1 in iterations: "
          , centroids[0])
    print("centroid No. 2 in iterations: "
          , centroids[1])
    print("centroid No. 3 in iterations: "
          , centroids[2])

    plt.show()
