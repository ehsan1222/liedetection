import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split


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


if __name__ == "__main__":
    x_data, y_data = get_data()
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, test_size=.25, random_state=0)
    