import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


def linear_regression(dataset, targetset, weights, alpha):
    new_weights = weights.copy()
    for i in range(0, len(weights)):
        correction = 0
        for j in range(0, len(dataset)):
            sample = dataset[j]
            predicted = predict_lr(sample, weights)
            expected = targetset[j]
            difference = expected - predicted
            if i == 0:
                correction += difference
            else:
                correction += difference * sample[i-1]
        new_weights[i] = weights[i] + alpha * correction
    return new_weights


def predict_lr(sample, weights):
    predicted = weights[0]
    for i in range(0, len(sample)):
        predicted += weights[i+1]*sample[i]
    return predicted


def main():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5

    weights = [0, 0, 0]
    for i in range(0, 10000):
        weights = linear_regression(X, y, weights, 0.0001)
        print(weights)

    n = len(X)
    mse = 0
    for i in range(0, n):
        sample = X[i]
        expected = y[i]
        predicted = round(predict_lr(sample, weights), 0)
        mse += math.pow(expected - predicted, 2)
        # print("expected: "+str(expected)+" predicted: "+str(predicted))
    mse /= n
    print("y=(" + str(round(weights[0], 2)) + ")+(" + str(round(weights[1], 2)) + ")x1+("+str(round(weights[2], 2))+")x2"+" mse: "+str(mse))

    plt.clf()
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.plot([x1_min, x1_max], [x2_min, x2_max], [weights[0]+weights[1]*x1_min+weights[2]*x2_min, (weights[0]+weights[1]*x1_max+weights[2]*x2_max)], color="k")

    plt.show()


main()
