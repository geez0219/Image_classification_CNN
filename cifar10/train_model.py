import os
import numpy as np
import tensorflow as tf

import pickle


def LeNet5(seed):
    initializer = tf.keras.initializers.glorot_uniform(seed=seed)
    input_tensor = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(filters=6, kernel_size=[5,5], activation=tf.nn.relu, kernel_initializer=initializer)(input_tensor)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=[5,5], activation=tf.nn.relu, kernel_initializer=initializer)(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=84, activation=tf.nn.relu, kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dense(units=120, activation=tf.nn.relu, kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, kernel_initializer=initializer)(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=x)
    return model


def LeNet5NoSeed():
    input_tensor = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(filters=6, kernel_size=[5,5], activation=tf.nn.relu)(input_tensor)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=[5,5], activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=84, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(units=120, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=x)
    return model


def VGG16(seed):
    initializer = tf.keras.initializers.glorot_uniform(seed=seed)
    input_tensor = tf.keras.layers.Input(shape=(32,32,3))

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu, kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu, kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, kernel_initializer=initializer)(x)

    return tf.keras.models.Model(inputs=input_tensor, outputs=x)


def VGG16NoSeed():
    input_tensor = tf.keras.layers.Input(shape=(32,32,3))

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(x)

    return tf.keras.models.Model(inputs=input_tensor, outputs=x)


def ResNet18():
    def ResNet_block(x, filters, mode):
        if mode == 0:  # the dense block size remain the same
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3,3], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)

            downstream = x

        else:  # the dense block size downsample by half
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same", strides=[2,2],
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)

            downstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same", strides=[2,2],
                                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        output = tf.keras.layers.add([upstream, downstream])
        output = tf.keras.layers.Activation("relu")(output)

        return output

    stack_num = 3
    filter_num = [16, 32, 64]
    depth = 2

    input_tensor = tf.keras.layers.Input(shape=[32, 32, 3])
    x = input_tensor
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=[3,3], padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    for stack_idx in range(stack_num):
        with tf.name_scope("stack{}".format(stack_idx)):
            for block_idx in range(depth):
                if stack_idx > 0 and block_idx == 0:
                    mode = 1
                else:
                    mode = 0
                x = ResNet_block(x, filters=filter_num[stack_idx], mode=mode)

    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(x)

    return tf.keras.models.Model(inputs=input_tensor, outputs=x)


def train_model(x_train, y_train, seed):
    tf.set_random_seed(seed)
    model = LeNet5(seed)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.categorical_accuracy])

    np.random.seed(seed)
    tf.set_random_seed(seed)
    model.fit(x_train, y_train, epochs=1, shuffle=True)
    return model


# define new Tensorboard to add learning rate
class XTensorboard(tf.keras.callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs.update({"lr": tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def lr_schedule(epoch):
    lr = 1e-3
    if epoch < 80:
        pass

    elif epoch < 120:
        lr *= 0.5

    elif epoch < 160:
        lr *= 0.5**2

    elif epoch < 180:
        lr *= 0.5**3

    else:
        lr *= 0.5**4

    return lr


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = pickle.load(open("dataset.pkl", "rb"))
    seed = 0

    with tf.Graph().as_default():
        sess = tf.Session()
        tf.keras.backend.set_session(sess)
        model = ResNet18()

        # model = tf.keras.models.load_model("Resnet18_checkpoint2.h5")
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.categorical_accuracy])

        tb_cb = XTensorboard(log_dir="tensorboard3")
        ck_cb = tf.keras.callbacks.ModelCheckpoint(filepath="Resnet18_checkpoint3.h5", save_best_only=True)
        ls_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        lr_cb = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                     cooldown=0,
                                                     patience=5,
                                                     min_lr=0.5e-6)

        # train with daya argumentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

        datagen.fit(x_train)

        model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                            validation_data=(x_test, y_test),
                            initial_epoch=0,
                            epochs=300,
                            verbose=1,
                            workers=4,
                            callbacks=[tb_cb, ck_cb, ls_cb, lr_cb])





















