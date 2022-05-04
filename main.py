import math
import random
import numpy
from sklearn import datasets


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
        for t in range(0, len(minibatch_dataset)):
            tmp_dataset.remove(minibatch_dataset[t])
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
        # to avoid math exception
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
    minibatch_dimension = 10
    precision = 2
    n_sample_per_label = 40
    decadence = 1 + learning_rate
    n_max_epoch = 1000
    normalized_dataset = dataset.copy()
    n_sample = len(dataset)
    random_set = True
    early_termination = False

    # computation of min and max values for each feature
    mins_maxs = []
    for i in range(0, n_features):
        mins_maxs.insert(i, (dataset[:, i].min(), dataset[:, i].max()))
    means = []
    std_devs = []
    for i in range(0, n_features):
        mean = 0
        std_dev = 0
        for sample in dataset:
            mean += sample[i]
        mean /= n_sample
        means.insert(i, mean)
        for sample in dataset:
            std_dev += math.pow(sample[i] - mean, 2)
        std_dev /= n_sample
        std_devs.insert(i, std_dev)

    # dataset normalization
    for i in range(0, n_sample):
        for j in range(0, n_features):
            # # min-max normalization
            # normalized_dataset[i][j] -= mins_maxs[j][0]
            # normalized_dataset[i][j] /= (mins_maxs[j][1] - mins_maxs[j][0])

            # z-mean normalization
            normalized_dataset[i][j] -= means[j]
            normalized_dataset[i][j] /= std_devs[j]

    # training and test samples extraction from dataset (random/non-random)
    if random_set is True:
        # data structure manipulation to allow a correct random sampling
        tmp_dataset = []
        for i in range(0, n_sample):
            tmp_sample = []
            for j in range(0, n_features):
                tmp_sample.insert(j, normalized_dataset[i][j])
            tmp_sample.insert(n_features, labels[i])
            tmp_dataset.insert(i, tmp_sample)

        # random training and test sets definition
        training_dataset = numpy.concatenate((numpy.concatenate(
            (random.sample(tmp_dataset[:50], n_sample_per_label),
             random.sample(tmp_dataset[50:100], n_sample_per_label)), axis=0),
                                              random.sample(tmp_dataset[100:150], n_sample_per_label)), axis=0)
        n_training = len(training_dataset)
        training_dataset_labels = []
        for i in range(0, n_training):
            training_dataset_labels.insert(i, training_dataset[i][n_features])

        test_dataset = tmp_dataset.copy()
        for i in range(0, n_training):
            for j in range(0, len(test_dataset)):
                training_sample = training_dataset[i]
                test_sample = test_dataset[j]
                equals = True
                # deep-equal
                for k in range(0, n_features):
                    if training_sample[k] != test_sample[k]:
                        equals = False
                        break
                if equals is True:
                    test_dataset.pop(j)
                    break

        n_test = len(test_dataset)
        test_dataset_labels = []
        for i in range(0, n_test):
            test_dataset_labels.insert(i, test_dataset[i][n_features])
    else:
        # non-random training and test sets definition
        training_dataset = numpy.concatenate((numpy.concatenate((normalized_dataset[0:40], normalized_dataset[50:90]),
                                                                axis=0), normalized_dataset[100:140]), axis=0)
        n_training = len(training_dataset)
        training_dataset_labels = numpy.concatenate((numpy.concatenate((labels[0:40], labels[50:90]),
                                                                       axis=0), labels[100:140]), axis=0)
        test_dataset = numpy.concatenate((numpy.concatenate((normalized_dataset[40:50], normalized_dataset[90:100]),
                                                            axis=0), normalized_dataset[140:150]), axis=0)
        n_test = len(test_dataset)
        test_dataset_labels = numpy.concatenate((numpy.concatenate((labels[40:50], labels[90:100]),
                                                                   axis=0), labels[140:150]), axis=0)

    # LINEAR REGRESSION
    linear_weights = numpy.zeros(n_features + 1)
    min_mse = mean_square_error(training_dataset, training_dataset_labels, linear_weights, n_features, precision)
    n_iterations = 0
    min_linear_weights = linear_weights
    n_min_epoch = 0
    while n_iterations < n_max_epoch:
        n_iterations += 1
        linear_weights = regression_minibatch(training_dataset, training_dataset_labels, linear_weights, n_features,
                                              learning_rate/(math.pow(decadence, n_iterations)),
                                              predict_linear, minibatch_dimension)
        # print(linear_weights)
        curr_mse = mean_square_error(training_dataset, training_dataset_labels, linear_weights, n_features, precision)
        if curr_mse <= min_mse:
            min_mse = curr_mse
            min_linear_weights = linear_weights
            n_min_epoch = n_iterations
        # early termination
        elif early_termination is True:
            break

    # statistics
    min_0, min_1, min_2 = 1, 1, 1
    max_0, max_1, max_2 = 0, 0, 0
    mean_0, mean_1, mean_2 = 0, 0, 0
    n_0, n_1, n_2 = 0, 0, 0
    for i in range(0, n_training):
        sample = training_dataset[i]
        expected = training_dataset_labels[i]
        predicted = predict_linear(sample, min_linear_weights, n_features)
        # print("expected: " + str(expected) + " predicted: " + str(predicted))
        if expected == 0:
            mean_0 += predicted
            n_0 += 1
            if predicted > max_0:
                max_0 = predicted
            if predicted < min_0:
                min_0 = predicted
        elif expected == 1:
            mean_1 += predicted
            n_1 += 1
            if predicted > max_1:
                max_1 = predicted
            if predicted < min_1:
                min_1 = predicted
        else:
            mean_2 += predicted
            n_2 += 1
            if predicted > max_2:
                max_2 = predicted
            if predicted < min_2:
                min_2 = predicted
    mean_0 /= n_0
    mean_1 /= n_1
    mean_2 /= n_2
    threshold_1 = (max_0 + min_1)/2
    threshold_2 = (max_1 + min_2)/2

    accuracy = 0
    for i in range(0, n_test):
        sample = test_dataset[i]
        expected = test_dataset_labels[i]
        predicted = round(predict_linear(sample, min_linear_weights, n_features), precision)
        # print("expected: " + str(expected) + " predicted: " + str(predicted))
        if predicted < threshold_1:
            predicted = 0
        elif predicted < threshold_2:
            predicted = 1
        else:
            predicted = 2
        if expected == predicted:
            accuracy += 1
    accuracy /= n_test
    accuracy *= 100

    print("LINEAR REGRESSION | y=(" + str(round(min_linear_weights[0], precision)) + ")+(" +
          str(round(min_linear_weights[1], precision)) + ")x1+(" + str(round(min_linear_weights[2], precision))
          + ")x2" + "+(" + str(round(min_linear_weights[3], precision)) + ")x3+(" +
          str(round(min_linear_weights[4], precision)) + ")x4" + " | mse: " + str(round(min_mse, precision)) +
          ", accuracy on test set: " + str(round(accuracy, precision)) + "%")
    print("labels 0's mean: " + str(round(mean_0, precision)) + ", labels 1's mean: " +
          str(round(mean_1, precision)) + ", labels 2's mean: " + str(round(mean_2, precision)) +
          ", \nlabels 0's max: " + str(round(max_0, precision))
          + ", labels 1's max: " + str(round(max_1, precision)) + ",labels 2's max: " +
          str(round(max_2, precision)) + ", \nlabels 0's min: " + str(round(min_0, precision)) +
          ", labels 1's min: " + str(round(min_1, precision)) + ", labels 2's min: "
          + str(round(min_2, precision)) + ", \nepochs: " + str(n_min_epoch) + ", threshold_1: "
          + str(round(threshold_1, precision)) + ", threshold_2: " + str(round(threshold_2, precision)) + ".\n")

    # training and test sets labels definition for logit
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

    # LOGISTIC REGRESSION
    for j in range(0, 3):
        logistic_weights = numpy.zeros(n_features + 1)
        min_bce = binary_cross_entropy(training_dataset, training_dataset_labels_i[j],
                                       logistic_weights, n_features, precision)
        min_logistic_weights = logistic_weights
        n_min_epoch = 0
        n_iterations = 0
        while n_iterations < n_max_epoch:
            n_iterations += 1
            logistic_weights = regression_minibatch(training_dataset, training_dataset_labels_i[j], logistic_weights,
                                                    n_features, learning_rate/(math.pow(decadence, n_iterations)),
                                                    predict_logistic, minibatch_dimension)
            # print(logistic_weights)
            curr_bce = binary_cross_entropy(training_dataset, training_dataset_labels_i[j], logistic_weights,
                                            n_features, precision)
            if curr_bce <= min_bce:
                min_bce = curr_bce
                min_logistic_weights = logistic_weights
                n_min_epoch = n_iterations
            # early termination
            elif early_termination is True:
                break

        pos_mean, neg_mean, n_pos, n_neg = 0, 0, 0, 0
        pos_max, neg_max = 0, 0
        pos_min, neg_min = 1, 1
        for i in range(0, n_training):
            sample = training_dataset[i]
            expected = training_dataset_labels_i[j][i]
            predicted = predict_logistic(sample, min_logistic_weights, n_features)
            # print("expected: " + str(expected) + " predicted: " + str(predicted))
            if expected == 1:
                pos_mean += predicted
                n_pos += 1
                if predicted > pos_max:
                    pos_max = predicted
                if predicted < pos_min:
                    pos_min = predicted
            else:
                neg_mean += predicted
                n_neg += 1
                if predicted > neg_max:
                    neg_max = predicted
                if predicted < neg_min:
                    neg_min = predicted
        pos_mean /= n_pos
        neg_mean /= n_neg
        threshold = (neg_max + pos_min)/2

        accuracy = 0
        for i in range(0, n_test):
            sample = test_dataset[i]
            expected = test_dataset_labels_i[j][i]
            predicted = round(predict_logistic(sample, min_logistic_weights, n_features), precision)
            # print("expected: " + str(expected) + " predicted: " + str(predicted))
            if predicted >= threshold:
                predicted = 1
            else:
                predicted = 0
            if expected == predicted:
                accuracy += 1
        accuracy /= n_test
        accuracy *= 100

        print("LOGISTIC REGRESSION | y=1/(1+e^-((" + str(round(min_logistic_weights[0], precision)) + ")+(" +
              str(round(min_logistic_weights[1], precision)) + ")x1+(" + str(round(min_logistic_weights[2], precision))
              + ")x2" + "+(" + str(round(min_logistic_weights[3], precision)) + ")x3+(" +
              str(round(min_logistic_weights[4], precision)) + ")x4))" + " | bce: " + str(round(min_bce, precision)) +
              ", accuracy on test set " + str(j) + ": " + str(round(accuracy, precision)) + "%")
        print("positive's mean: " + str(round(pos_mean, precision)) + ", negative's mean: " +
              str(round(neg_mean, precision)) + ", \npositive's max: " + str(round(pos_max, precision))
              + ", negative's max: " + str(round(neg_max, precision)) + ", \npositive's min: " +
              str(round(pos_min, precision)) + ", negative's min: " + str(round(neg_min, precision)) +
              ", \nepochs: " + str(n_min_epoch) + ", threshold: " + str(round(threshold, precision)) + ".\n")


main()
