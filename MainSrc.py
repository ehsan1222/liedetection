import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from myClassifiers import *


def transpose(data):
    return data.transpose()


def get_data():
    """get feature extracted data and transpose features -- x(1342 * 17175) y(1342 * 1)"""
    with h5py.File("feature_extracts/eeg_X.h5", 'r') as f:
        x = f['/extractor'][()]
    with h5py.File("feature_extracts/eeg_y.h5", 'r') as f:
        y = f['/extractor'][()]
    with h5py.File("feature_extracts/eeg_means.h5", 'r') as f:
        means = f['/extractor'][()]
    with h5py.File("feature_extracts/eeg_X_ben.h5", "r") as f:
        x_ben = f["/extractor"][()]
    with h5py.File("feature_extracts/eeg_y_ben.h5", 'r') as f:
        y_ben = f['/extractor'][()]
    with h5py.File("feature_extracts/eeg_y_ben.h5") as f:
        means_bean = f['/extractor'][()]

    x = transpose(x)
    x_ben = transpose(x_ben)
    x_combined = np.concatenate((x, x_ben), axis=0)
    y_combined = np.concatenate((y, y_ben), axis=0)

    return x_combined, y_combined


def cross_validation(y_predict, y_true, classification_name):
    correct_count = 0
    true_pos_count = 0
    false_pos_count = 0
    true_neg_count = 0
    false_neg_count = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            correct_count += 1
            if y_predict[i] == 0:
                true_pos_count += 1
            else:
                true_neg_count += 1
        else:
            if y_predict[i] == 0:
                false_pos_count += 1
            else:
                false_neg_count += 1
    root_mean_squared_error = np.sqrt(mean_squared_error(y_true, y_predict))
    validate_data = {
        "correct": correct_count,
        "true_pos": true_pos_count,
        "false_pos": false_pos_count,
        "true_neg": true_neg_count,
        "false_neg": false_neg_count,
        "rmse": root_mean_squared_error,
        "alg": classification_name
    }
    return validate_data


def select_thee_best_of_classifiers(list_of_validate_classifiers):
    first_classifier_value = 0
    first_classifier_validate = None
    second_classifier_value = 0
    second_classifier_validate = None
    third_classifier_value = 0
    third_classifier_validate = None

    for classifier_validation in list_of_validate_classifiers:
        if classifier_validation["correct"] > first_classifier_value:
            third_classifier_value = second_classifier_value
            third_classifier_validate = second_classifier_validate
            second_classifier_value = first_classifier_value
            second_classifier_validate = first_classifier_validate
            first_classifier_value = classifier_validation["correct"]
            first_classifier_validate = classifier_validation
        elif classifier_validation["correct"] > second_classifier_value:
            third_classifier_value = second_classifier_value
            third_classifier_validate = second_classifier_validate
            second_classifier_value = classifier_validation["correct"]
            second_classifier_validate = classifier_validation
        elif classifier_validation["correct"] > third_classifier_value:
            third_classifier_value = classifier_validation["correct"]
            third_classifier_validate = classifier_validation

    return [first_classifier_validate, second_classifier_validate, third_classifier_validate]


if __name__ == "__main__":
    # get eeg feature extracted data
    x_data, y_data = get_data()

    # divide data to train and test with randomised algorithms
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.25, random_state=0)

    # classification
    y_predict_svm = support_vector_machine_classifier(x_train, y_train, x_test)
    y_predict_nb = naive_bayes_classifier(x_train, y_train, x_test)
    y_predict_knn = k_nearest_neighbor_classifier(x_train, y_train, x_test)
    y_predict_mlp = multi_layer_perceptron_classifier(x_train, y_train, x_test)
    y_predict_lda = linear_discriminant_analysis_classifier(x_train, y_train, x_test)

    # select three best classifiers
    # get classification accuracy
    c_validate_svm = cross_validation(y_predict_svm, y_test, "svm")
    c_validate_nb = cross_validation(y_predict_nb, y_test, "nb")
    c_validate_knn = cross_validation(y_predict_knn, y_test, "knn")
    c_validate_mlp = cross_validation(y_predict_mlp, y_test, "mlp")
    c_validate_lda = cross_validation(y_predict_lda, y_test, "lda")

    # select three best of classifiers
    list_of_validate_classifiers = [
        c_validate_svm,
        c_validate_nb,
        c_validate_knn,
        c_validate_mlp,
        c_validate_lda
    ]
    list_of_three_best_validate_classifiers = select_thee_best_of_classifiers(list_of_validate_classifiers)
    for item in list_of_three_best_validate_classifiers:
        print(item)