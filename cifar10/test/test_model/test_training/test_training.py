import numpy as np
import tensorflow as tf

from cifar10.train_model import VGG16NoSeed


# def test_no_shuffle(load_data):
#     """
#     try to train two identical models control randomness of:
#     1. weight initialization
#     """
#     (x_train, y_train), (x_test, y_test) = load_data
#
#     seed = np.random.random_integers(0, 100000)
#     with tf.Graph().as_default():
#         # crucial for weight initialization
#         tf.set_random_seed(seed)
#
#         # need to force system use CPU and disable CPU multi-thread process
#         config = tf.ConfigProto(
#             device_count={"GPU": 0},
#             intra_op_parallelism_threads=1,
#             inter_op_parallelism_threads=1)
#         sess = tf.Session(config=config)
#         tf.keras.backend.set_session(session=sess)
#
#         model = LeNet5NoSeed()
#         model.compile(loss=tf.keras.losses.categorical_crossentropy,
#                       optimizer=tf.keras.optimizers.Adam(),
#                       metrics=[tf.keras.metrics.categorical_accuracy])
#         model.fit(x_train, y_train, epochs=3, shuffle=False)
#
#     model_weight = model.get_weights()
#
#     with tf.Graph().as_default():
#         # crucial for weight initialization
#         tf.set_random_seed(seed)
#
#         # need to disable CPU multi-thread process
#         config = tf.ConfigProto(
#             device_count={"GPU": 0},
#             intra_op_parallelism_threads=1,
#             inter_op_parallelism_threads=1)
#         sess = tf.Session(config=config)
#         tf.keras.backend.set_session(session=sess)
#
#         model2 = LeNet5NoSeed()
#         model2.compile(loss=tf.keras.losses.categorical_crossentropy,
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=[tf.keras.metrics.categorical_accuracy])
#         model2.fit(x_train, y_train, epochs=3, shuffle=False)
#     model_weight2 = model2.get_weights()
#
#     for i in range(len(model_weight)):
#         assert np.array_equal(model_weight[i], model_weight2[i])
#
#
# def test_shuffle(load_data):
#     """
#     try to train two identical models control randomness of:
#     1. weight initialization
#     2. training calculation
#     3. training shuffling
#     """
#     (x_train, y_train), (x_test, y_test) = load_data
#
#     seed = np.random.random_integers(0, 100000)
#
#     with tf.Graph().as_default():
#         np.random.seed(seed)  # crucial for shuffling
#         tf.set_random_seed(seed)  # crucial for weight initialization
#
#         # need to force system use CPU and disable CPU multi-thread process
#         config = tf.ConfigProto(
#             device_count={"GPU": 0},
#             intra_op_parallelism_threads=1,
#             inter_op_parallelism_threads=1)
#         sess = tf.Session(config=config)
#         tf.keras.backend.set_session(session=sess)
#
#         model = LeNet5NoSeed()
#         model.compile(loss=tf.keras.losses.categorical_crossentropy,
#                       optimizer=tf.keras.optimizers.Adam(),
#                       metrics=[tf.keras.metrics.categorical_accuracy])
#
#         model.fit(x_train, y_train, epochs=3, shuffle=True)
#
#     model_weight = model.get_weights()
#
#     with tf.Graph().as_default():
#         np.random.seed(seed)  # crucial for shuffling
#         tf.set_random_seed(seed)  # crucial for weight initialization
#
#         # need to disable CPU multi-thread process
#         config = tf.ConfigProto(
#             device_count={"GPU": 0},
#             intra_op_parallelism_threads=1,
#             inter_op_parallelism_threads=1)
#         sess = tf.Session(config=config)
#         tf.keras.backend.set_session(session=sess)
#
#         model2 = LeNet5NoSeed()
#         model2.compile(loss=tf.keras.losses.categorical_crossentropy,
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=[tf.keras.metrics.categorical_accuracy])
#
#         model2.fit(x_train, y_train, epochs=3, shuffle=True)
#     model_weight2 = model2.get_weights()
#
#     for i in range(len(model_weight)):
#         assert np.array_equal(model_weight[i], model_weight2[i])


def test_shuffle_dropout_batch_normalization(load_data):
    """
    try to train two identical models control randomness of:
    1. weight initialization
    2. training calculation
    3. training shuffling
    4. dropout, batch_normalization
    """
    (x_train, y_train), (x_test, y_test) = load_data

    seed = np.random.random_integers(0, 100000)

    with tf.Graph().as_default():
        np.random.seed(seed)  # crucial for shuffling
        tf.set_random_seed(seed)  # crucial for weight initialization

        # need to force system use CPU and disable CPU multi-thread process
        config = tf.ConfigProto(
            device_count={"GPU": 0},
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(session=sess)

        model = VGG16NoSeed()
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.categorical_accuracy])

        model.fit(x_train, y_train, epochs=2, shuffle=True)

    model_weight = model.get_weights()

    with tf.Graph().as_default():
        np.random.seed(seed)  # crucial for shuffling
        tf.set_random_seed(seed)  # crucial for weight initialization

        # need to disable CPU multi-thread process
        config = tf.ConfigProto(
            device_count={"GPU": 0},
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(session=sess)

        model2 = VGG16NoSeed()
        model2.compile(loss=tf.keras.losses.categorical_crossentropy,
                       optimizer=tf.keras.optimizers.Adam(),
                       metrics=[tf.keras.metrics.categorical_accuracy])

        model2.fit(x_train, y_train, epochs=2, shuffle=True)
    model_weight2 = model2.get_weights()

    for i in range(len(model_weight)):
        assert np.array_equal(model_weight[i], model_weight2[i])