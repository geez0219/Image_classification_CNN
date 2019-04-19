import tensorflow as tf
import pickle


def generate_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train/255.0
    x_test = x_test/255.0

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    pickle.dump(((x_train, y_train), (x_test, y_test)), open("dataset.pkl", "wb"))

