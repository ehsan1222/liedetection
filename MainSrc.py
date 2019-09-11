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


def cross_validation(y_predict, y_true, classifier_name):
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
        "alg": classifier_name,
        "predict": y_predict
    }
    return validate_data


def cross_validation2(y_predict, y_true, classifier_name):
    correct_count = 0
    true_pos_count = 0
    false_pos_count = 0
    true_neg_count = 0
    false_neg_count = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            correct_count += 1
            if y_predict[i] == -1:
                true_pos_count += 1
            else:
                true_neg_count += 1
        else:
            if y_predict[i] == -1:
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
        "alg": classifier_name,
        "predict": y_predict
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


def stacking(list_of_three_best_validate_classifiers, x_next_step_train, y_next_step_train, x_test):
    # change 0 input to -1
    list_of_three_best_validate_classifiers[0]["predict"][
        list_of_three_best_validate_classifiers[0]["predict"] == 0] = -1
    list_of_three_best_validate_classifiers[1]["predict"][
        list_of_three_best_validate_classifiers[1]["predict"] == 0] = -1
    list_of_three_best_validate_classifiers[2]["predict"][
        list_of_three_best_validate_classifiers[2]["predict"] == 0] = -1
    y_next_step_train_tmp = np.array(y_next_step_train)
    y_next_step_train_tmp[y_next_step_train_tmp == 0] = -1
    # train mlp
    mlp_input = transpose(np.array([list_of_three_best_validate_classifiers[0]["predict"],
                                    list_of_three_best_validate_classifiers[1]["predict"],
                                    list_of_three_best_validate_classifiers[2]["predict"]]))
    mlp = MLPClassifier(hidden_layer_sizes=(5, 3), activation="logistic", max_iter=1000)
    mlp.fit(mlp_input, transpose(np.array(y_next_step_train_tmp)))

    # get three best classifier's name and train three best classifiers with next step train data
    predict_1 = select_classifier(list_of_three_best_validate_classifiers[0]["alg"], x_next_step_train,
                                  y_next_step_train, x_test)
    predict_2 = select_classifier(list_of_three_best_validate_classifiers[1]["alg"], x_next_step_train,
                                  y_next_step_train, x_test)
    predict_3 = select_classifier(list_of_three_best_validate_classifiers[2]["alg"], x_next_step_train,
                                  y_next_step_train, x_test)
    # list of predict three best classifiers
    predict_list = [predict_1, predict_2, predict_3]
    # transpose data for stacking
    stacking_predict = transpose(np.array(predict_list))
    # get predict values to mlp to get pattern incorrect
    stacking_predict[stacking_predict == 0] = -1
    return mlp.predict(stacking_predict)


def select_classifier(classifier_name, x_next_step_train, y_next_step_train, x_test):
    if classifier_name == "svm":
        return support_vector_machine_classifier(x_next_step_train, y_next_step_train, x_test)
    elif classifier_name == "nb":
        return naive_bayes_classifier(x_next_step_train, y_next_step_train, x_test)
    elif classifier_name == "knn":
        return k_nearest_neighbor_classifier(x_next_step_train, y_next_step_train, x_test)
    elif classifier_name == "mlp":
        return multi_layer_perceptron_classifier(x_next_step_train, y_next_step_train, x_test)
    elif classifier_name == "lda":
        return linear_discriminant_analysis_classifier(x_next_step_train, y_next_step_train, x_test)
    else:
        return None


if __name__ == "__main__":
    # get eeg feature extracted data
    x_data, y_data = get_data()

    # divide data to train and test with randomised algorithms
    x_train, x_next_step, y_train, y_next_step = train_test_split(x_data, y_data, test_size=.5, random_state=0)
    x_next_step_train, x_test, y_next_step_train, y_test = train_test_split(x_next_step, y_next_step, test_size=.4,
                                                                            random_state=0)

    # classification
    y_predict_svm = support_vector_machine_classifier(x_train, y_train, x_next_step_train)
    y_predict_nb = naive_bayes_classifier(x_train, y_train, x_next_step_train)
    y_predict_knn = k_nearest_neighbor_classifier(x_train, y_train, x_next_step_train)
    y_predict_mlp = multi_layer_perceptron_classifier(x_train, y_train, x_next_step_train)
    y_predict_lda = linear_discriminant_analysis_classifier(x_train, y_train, x_next_step_train)

    # select three best classifiers
    # get classification accuracy
    c_validate_svm = cross_validation(y_predict_svm, y_next_step_train, "svm")
    print(f"SVM: {(c_validate_svm['correct'] / len(y_next_step_train)) * 100} %")
    c_validate_nb = cross_validation(y_predict_nb, y_next_step_train, "nb")
    print(f"Naive Bayes:{(c_validate_nb['correct'] / len(y_next_step_train)) * 100} %")
    c_validate_knn = cross_validation(y_predict_knn, y_next_step_train, "knn")
    print(f"KNN:{(c_validate_knn['correct'] / len(y_next_step_train)) * 100} %")
    c_validate_mlp = cross_validation(y_predict_mlp, y_next_step_train, "mlp")
    print(f"MLP:{(c_validate_mlp['correct'] / len(y_next_step_train)) * 100} %")
    c_validate_lda = cross_validation(y_predict_lda, y_next_step_train, "lda")
    print(f"LDA:{(c_validate_lda['correct'] / len(y_next_step_train)) * 100} %")
    # select three best of classifiers
    list_of_validate_classifiers = [
        c_validate_svm,
        c_validate_nb,
        c_validate_knn,
        c_validate_mlp,
        c_validate_lda
    ]
    list_of_three_best_validate_classifiers = select_thee_best_of_classifiers(list_of_validate_classifiers)

    # create a new predict item from three best of classifier results
    final_predict = stacking(list_of_three_best_validate_classifiers, x_next_step_train, y_next_step_train, x_test)
    y_test[y_test == 0] = -1
    final_validation = cross_validation2(final_predict, y_test, "stacking")
    print(f"RMSE: {final_validation['rmse']}")
    print(f"{(final_validation['correct'] / len(y_test)) * 100} %")
