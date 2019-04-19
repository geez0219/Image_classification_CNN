import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['PYTHONHASHSEED'] = "0"
import numpy as np
import tensorflow as tf
from cifar10.train_model import train_model


# def test_weight_initial():
#     """
#     test whether the weight can be initialized reproducible
#     """
#     seed = 0
#     model1 = VGG16(seed)
#     model2 = VGG16(seed)
#     weight_list1 = model1.get_weights()
#     weight_list2 = model2.get_weights()
#
#     assert len(weight_list1) == len(weight_list2)
#
#     for i in range(len(weight_list1)):
#         assert np.array_equal(weight_list1[i], weight_list2[i])


# def test_train_without_shuffle(load_data):
#
#     seed = 0
#     np.random.seed(seed)
#     tf.set_random_seed(seed)
#
#     (x_train, y_train), (x_test, y_test) = load_data
#
#     model1 = LeNet5(seed)
#     model2 = LeNet5(seed)
#     weight_list1 = model1.get_weights()
#     weight_list2 = model2.get_weights()
#
#     assert len(weight_list1) == len(weight_list2)
#
#     for i in range(len(weight_list1)):
#         assert np.array_equal(weight_list1[i], weight_list2[i])
#
#     model1.compile(loss=tf.keras.losses.categorical_crossentropy,
#                    optimizer=tf.keras.optimizers.Adam(),
#                    metrics=[tf.keras.metrics.categorical_accuracy])
#
#     model2.compile(loss=tf.keras.losses.categorical_crossentropy,
#                    optimizer=tf.keras.optimizers.Adam(),
#                    metrics=[tf.keras.metrics.categorical_accuracy])
#
#     model1.fit(x_train, y_train, epochs=1, shuffle=False)
#     model2.fit(x_train, y_train, epochs=1, shuffle=False)
#
#     weight_list1 = model1.get_weights()
#     weight_list2 = model2.get_weights()
#
#     assert len(weight_list1) == len(weight_list2)
#
#     for i in range(len(weight_list1)):
#         assert np.array_equal(weight_list1[i], weight_list2[i])


def test_reproduce(load_data):
    (x_train, y_train), (x_test, y_test) = load_data
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    seed = 0
    model1 = train_model(x_train, y_train, seed)
    model2 = train_model(x_train, y_train, seed)

    weight_list1 = model1.get_weights()
    weight_list2 = model2.get_weights()

    assert len(weight_list1) == len(weight_list2)

    for i in range(len(weight_list1)):
        assert np.array_equal(weight_list1[i], weight_list2[i])


