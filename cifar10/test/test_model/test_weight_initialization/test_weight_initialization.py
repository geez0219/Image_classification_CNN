import numpy as np
import tensorflow as tf

from cifar10.train_model import LeNet5, LeNet5NoSeed


def test_with_seed():
    """
    using kernel initializer with seed
    """
    seed = np.random.random_integers(0, 100000)
    with tf.Graph().as_default():
        sess = tf.Session()
        tf.keras.backend.set_session(session=sess)
        model = LeNet5(seed)
    model_weight = model.get_weights()

    with tf.Graph().as_default():
        sess = tf.Session()
        tf.keras.backend.set_session(session=sess)
        model2 = LeNet5(seed)
    model_weight2 = model2.get_weights()

    for i in range(len(model_weight)):
        assert np.array_equal(model_weight[i], model_weight2[i])


# def test_without_seed():
#     """
#     suppose to fail
#     """
#     with tf.Graph().as_default():
#         sess = tf.Session()
#         tf.keras.backend.set_session(session=sess)
#         model = LeNet5NoSeed()
#     model_weight = model.get_weights()
#
#     with tf.Graph().as_default():
#         sess = tf.Session()
#         tf.keras.backend.set_session(session=sess)
#         model2 = LeNet5NoSeed()
#     model_weight2 = model2.get_weights()
#
#     for i in range(len(model_weight)):
#         assert np.array_equal(model_weight[i], model_weight2[i])

def test_without_seed():
    """
    set tf.random_set(seed)
    """
    seed = np.random.random_integers(0, 100000)
    with tf.Graph().as_default():
        # crucial for weight initialization
        tf.set_random_seed(seed)

        sess = tf.Session()
        tf.keras.backend.set_session(session=sess)
        model = LeNet5NoSeed()
    model_weight = model.get_weights()

    with tf.Graph().as_default():
        # crucial for weight initialization
        tf.set_random_seed(seed)

        sess = tf.Session()
        tf.keras.backend.set_session(session=sess)
        model2 = LeNet5NoSeed()
    model_weight2 = model2.get_weights()

    for i in range(len(model_weight)):
        assert np.array_equal(model_weight[i], model_weight2[i])
