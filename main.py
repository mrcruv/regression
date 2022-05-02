import math
import random

import numpy
from sklearn import datasets


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def regression(dataset, labels, weights, n_features, learning_rate, predict_function):
    # loss is mean square error loss for linear regression,
    # binary cross entropy loss for logistic regression
    # => same update rule
    new_weights = weights.copy()
    for i in range(0, n_features+1):
        correction = 0
        for j in range(0, len(dataset)):
            sample = dataset[j]
            predicted = predict_function(sample, weights, n_features)
            expected = labels[j]
            difference = expected - predicted
            if i == 0:
                correction += difference
            else:
                correction += difference * sample[i - 1]
        new_weights[i] += learning_rate * correction

    return new_weights


def regression_minibatch(dataset, labels, weights, n_features, learning_rate, predict_function, minibatch_dimension):
    new_weights = weights.copy()
    n_sample = len(dataset)

    tmp_dataset = []
    for i in range(0, n_sample):
        tmp_sample = []
        for j in range(0, n_features):
            tmp_sample.insert(j, dataset[i][j])
        tmp_sample.insert(n_features, labels[i])
        tmp_dataset.insert(i, tmp_sample)

    for i in range(0, int(n_sample/minibatch_dimension)):
        minibatch_dataset = random.sample(tmp_dataset, minibatch_dimension)
        for j in range(0, len(weights)):
            correction = 0
            for k in range(0, len(minibatch_dataset)):
                sample = minibatch_dataset[k]
                predicted = predict_function(sample, new_weights, n_features)
                expected = minibatch_dataset[k][n_features]
                difference = expected - predicted
                if k == 0:
                    correction += difference
                else:
                    correction += difference * sample[j - 1]
            new_weights[j] += learning_rate * correction

    return new_weights


def predict_logistic(sample, weights, n_features):
    x = weights[0]
    for i in range(0, n_features):
        x += weights[i + 1] * sample[i]
    predicted = sigmoid(x)
    return predicted


def predict_linear(sample, weights, n_features):
    predicted = weights[0]
    for i in range(0, n_features):
        predicted += weights[i + 1] * sample[i]
    return predicted


def mean_square_error(dataset, labels, weights, n_features, precision=2):
    res = 0
    n_sample = len(dataset)
    for i in range(0, n_sample):
        sample = dataset[i]
        expected = labels[i]
        predicted = round(predict_linear(sample, weights, n_features), precision)
        res += math.pow(expected - predicted, 2)
        # print("expected: " + str(expected)+" predicted: " + str(predicted))
    res /= n_sample
    return res


def binary_cross_entropy(dataset, labels, weights, n_features, precision=2):
    res = 0
    n_sample = len(dataset)
    for i in range(0, n_sample):
        sample = dataset[i]
        expected = labels[i]
        predicted = round(predict_logistic(sample, weights, n_features), precision)
        if predicted == 0:
            a = -math.inf
            b = math.log(1 - predicted, math.e)
        elif predicted == 1:
            a = math.log(predicted, math.e)
            b = -math.inf
        else:
            a = math.log(predicted, math.e)
            b = math.log(1 - predicted, math.e)
        res += -(expected*a + (1-expected)*b)
    res /= n_sample
    return res


def main():
    numpy.seterr(all="ignore")
    iris = datasets.load_iris()
    n_features = 4
    dataset = iris.data[:, :n_features]
    labels = iris.target
    learning_rate = 0.001
    minibatch_dimension = 12
    precision = 3

    training_dataset = numpy.concatenate((numpy.concatenate((dataset[0:40], dataset[50:90]), axis=0),
                                          dataset[100:140]), axis=0)
    training_dataset_labels = numpy.concatenate((numpy.concatenate((labels[0:40], labels[50:90]), axis=0),
                                                 labels[100:140]), axis=0)
    test_dataset = numpy.concatenate((numpy.concatenate((dataset[40:50], dataset[90:100]), axis=0),
                                      dataset[140:150]), axis=0)
    test_dataset_labels = numpy.concatenate((numpy.concatenate((labels[40:50], labels[90:100]), axis=0),
                                             labels[140:150]), axis=0)
    n_test = len(test_dataset)
    n_training = len(training_dataset)

    linear_weights = numpy.zeros(n_features + 1)
    min_mse = mean_square_error(training_dataset, training_dataset_labels, linear_weights, n_features, precision)
    n_iterations = 0
    while True:
        n_iterations += 1
        # linear_weights = regression(training_dataset, training_dataset_labels, linear_weights, n_features,
        # learning_rate, predict_linear)
        linear_weights = regression_minibatch(training_dataset, training_dataset_labels, linear_weights, n_features,
                                              learning_rate, predict_linear, minibatch_dimension)
        # print(linear_weights)
        curr_mse = mean_square_error(training_dataset, training_dataset_labels, linear_weights, n_features, precision)
        if curr_mse <= min_mse:
            min_mse = curr_mse
        else:
            break
    mse = min_mse

    accuracy = 0
    for i in range(0, n_test):
        sample = test_dataset[i]
        expected = test_dataset_labels[i]
        predicted = round(predict_linear(sample, linear_weights, n_features), 0)
        if expected == predicted:
            accuracy += 1
        # print("expected: " + str(expected)+" predicted: " + str(predicted))
    accuracy /= n_test
    accuracy *= 100
    accuracy = round(accuracy, precision)

    print("linear regression || y=(" + str(round(linear_weights[0], precision)) + ")+(" +
          str(round(linear_weights[1], precision)) + ")x1+(" + str(round(linear_weights[precision], precision))+")x2" +
          "+(" + str(round(linear_weights[3], precision)) + ")x3+(" + str(round(linear_weights[4], precision))+")x4" +
          " mse: " + str(round(mse, precision)) + ", accuracy on test set: " + str(round(accuracy, precision)) + "%")

    training_dataset_labels1 = numpy.copy(training_dataset_labels)
    test_dataset_labels1 = numpy.copy(test_dataset_labels)
    for i in range(0, n_training):  # 010
        if training_dataset_labels1[i] == 2:
            training_dataset_labels1[i] = 0
    for i in range(0, n_test):  # 010
        if test_dataset_labels1[i] == 2:
            test_dataset_labels1[i] = 0

    training_dataset_labels2 = numpy.copy(training_dataset_labels)
    test_dataset_labels2 = numpy.copy(test_dataset_labels)
    for i in range(0, n_training):  # 100
        if training_dataset_labels2[i] == 0:
            training_dataset_labels2[i] = 1
        else:
            training_dataset_labels2[i] = 0
    for i in range(0, n_test):  # 100
        if test_dataset_labels2[i] == 0:
            test_dataset_labels2[i] = 1
        else:
            test_dataset_labels2[i] = 0

    training_dataset_labels3 = numpy.copy(training_dataset_labels)
    test_dataset_labels3 = numpy.copy(test_dataset_labels)
    for i in range(0, n_training):  # 001
        if training_dataset_labels3[i] == 2:
            training_dataset_labels3[i] = 1
        else:
            training_dataset_labels3[i] = 0
    for i in range(0, n_test):  # 001
        if test_dataset_labels3[i] == 2:
            test_dataset_labels3[i] = 1
        else:
            test_dataset_labels3[i] = 0

    training_dataset_labels_i = [training_dataset_labels1, training_dataset_labels2, training_dataset_labels3]
    test_dataset_labels_i = [test_dataset_labels1, test_dataset_labels2, test_dataset_labels3]

    for j in range(0, 3):
        logistic_weights = numpy.zeros(n_features + 1)
        min_bce = binary_cross_entropy(training_dataset, training_dataset_labels_i[j],
                                       logistic_weights, n_features, precision)
        n_iterations = 0
        while True:
            n_iterations += 1
            logistic_weights = regression_minibatch(training_dataset, training_dataset_labels_i[j], logistic_weights,
                                                    n_features, learning_rate, predict_logistic, minibatch_dimension)
            # print(logistic_weights)
            curr_bce = binary_cross_entropy(training_dataset, training_dataset_labels_i[j], logistic_weights,
                                            n_features, precision)
            if curr_bce <= min_bce:
                min_bce = curr_bce
            else:
                break
        bce = min_bce

        pos_mean, neg_mean, n_pos, n_neg = 0, 0, 0, 0
        for i in range(0, n_training):
            sample = training_dataset[i]
            expected = training_dataset_labels_i[j][i]
            predicted = predict_logistic(sample, logistic_weights, n_features)
            # print("expected: " + str(expected) + " predicted: " + str(predicted))
            if expected == 1:
                pos_mean += predicted
                n_pos += 1
            else:
                neg_mean += predicted
                n_neg += 1
        pos_mean /= n_pos
        neg_mean /= n_neg
        threshold = pos_mean - neg_mean
        print("positive's mean: " + str(round(pos_mean, precision)) + ", negative's mean: " +
              str(round(neg_mean, precision)) + ", threshold: " + str(round(threshold, precision)))

        accuracy = 0
        for i in range(0, n_test):
            sample = test_dataset[i]
            expected = test_dataset_labels_i[j][i]
            predicted = round(predict_logistic(sample, logistic_weights, n_features), precision)
            # print("expected: " + str(expected) + " predicted: " + str(predicted))
            if predicted >= threshold:
                predicted = 1
            else:
                predicted = 0
            if expected == round(predicted, 0):
                accuracy += 1
        accuracy /= n_test
        accuracy *= 100
        accuracy = round(accuracy, precision)
        print("logistic regression || y=1/(1+e^-((" + str(round(logistic_weights[0], precision)) + ")+(" +
              str(round(logistic_weights[1], precision)) + ")x1+(" + str(round(logistic_weights[precision], precision))
              + ")x2" + "+(" + str(round(logistic_weights[3], precision)) + ")x3+(" +
              str(round(logistic_weights[4], precision)) + ")x4))" + " bce: " + str(round(bce, precision)) +
              ", accuracy on test set " + str(j) + ": " + str(round(accuracy, precision)) + "%")


main()
