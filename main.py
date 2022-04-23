import math

import matplotlib.pyplot as plt
from sklearn import datasets


def linear_regression(dataset, targetset, weights, alpha):
    new_weights = weights.copy()
    for i in range(0, len(weights)):
        correction = 0
        for j in range(0, len(dataset)):
            sample = dataset[j]
            predicted = predict_lr(sample, weights)
            expected = targetset[j][0]
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
    X = iris.data[:, 0:1]
    y = iris.data[:, 1:2]
    # y = iris.target

    weights = [0, 0]
    new_weights = linear_regression(X, y, weights, 0.1)
    for i in range(0, 10000):
        new_weights = linear_regression(X, y, new_weights, 0.0001)
        print(new_weights)

    n = len(X)
    variance = 0
    for i in range(0, n):
        sample = X[i]
        petal_length = sample[0]
        petal_width = y[i][0]
        predicted = predict_lr(sample, new_weights)
        variance += math.fabs(petal_width - predicted)
        print("petal lenght: "+str(petal_length)+" petal_width: "+str(petal_width)+" predicted: "+str(predicted))
    variance /= n
    print("y=("+str(round(new_weights[0], 2))+")+("+str(round(new_weights[1], 2))+")x "+"variance: "+str(variance))

    # x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    # y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x_min, x_max = X.min() - 0.5, X.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.scatter(X[:, 0], y, c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.plot([x_min, x_max], [new_weights[0]+new_weights[1]*x_min, new_weights[0]+new_weights[1]*x_max], color="k")

    plt.show()


main()
