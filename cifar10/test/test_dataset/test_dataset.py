import pickle

import numpy as np
import tensorflow as tf

from cifar10.generate_dataset import generate_dataset


def test_generate_dataset():
    generate_dataset()
    (x_train, y_train), (x_test, y_test) = pickle.load(open("dataset.pkl", "rb"))

    (x_train_exp, y_train_exp), (x_test_exp, y_test_exp) = tf.keras.datasets.cifar10.load_data()
    x_train_exp = x_train_exp/255.0
    x_test_exp = x_test_exp/255.0
    y_train_exp = tf.keras.utils.to_categorical(y_train_exp)
    y_test_exp = tf.keras.utils.to_categorical(y_test_exp)

    assert np.array_equal(x_train, x_train_exp)
    assert np.array_equal(y_train, y_train_exp)
    assert np.array_equal(x_test, x_test_exp)
    assert np.array_equal(y_test, y_test_exp)

