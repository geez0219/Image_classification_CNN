import tensorflow as tf
import numpy as np
import pickle


def VGG16():
    input_tensor = tf.keras.layers.Input(shape=(64,64,3))

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
    x = tf.keras.layers.Dense(units=200, activation=tf.nn.softmax)(x)

    return tf.keras.models.Model(inputs=input_tensor, outputs=x)


def ResNet20():
    def ResNet_block(x, filters, mode):
        if mode == 0:  # the dense block size remain the same
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same",
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
    filter_num = [32, 64, 128]
    depth = 3

    input_tensor = tf.keras.layers.Input(shape=[64, 64, 3])
    x = input_tensor
    x = tf.keras.layers.Conv2D(filters=filter_num[0], kernel_size=[3,3], padding="same",
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

    x = tf.keras.layers.AveragePooling2D(pool_size=16)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=200, activation=tf.nn.softmax)(x)

    return tf.keras.models.Model(inputs=input_tensor, outputs=x)


def ResNet50():
    def ResNet_block(x, filters, mode):
        if mode == 0:  # the dense block size remain the same
            # first part
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[1, 1], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            # second part
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            # third part
            upstream = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=[1, 1], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)

            # downstream
            downstream = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=[1, 1], padding="same",
                                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        else:  # the dense block size downsample by half
            # first part
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[1, 1], padding="same", strides=[2,2],
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            # second part
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            # third part
            upstream = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=[1, 1], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)

            # downstream
            downstream = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=[1, 1], padding="same", strides=[2,2],
                                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        output = tf.keras.layers.add([upstream, downstream])
        output = tf.keras.layers.Activation("relu")(output)

        return output

    stack_num = 4
    filter_num = [32, 64, 128, 128]
    depth = [3, 4, 6, 3]

    input_tensor = tf.keras.layers.Input(shape=[64, 64, 3])
    x = input_tensor
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    for stack_idx in range(stack_num):
        with tf.name_scope("stack{}".format(stack_idx)):
            for block_idx in range(depth[stack_idx]):
                if stack_idx > 0 and block_idx == 0:
                    mode = 1
                else:
                    mode = 0
                x = ResNet_block(x, filters=filter_num[stack_idx], mode=mode)

    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=200, activation=tf.nn.softmax)(x)

    return tf.keras.models.Model(inputs=input_tensor, outputs=x)


# define new Tensorboard to add learning rate
class XTensorboard(tf.keras.callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs.update({"lr": tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


if __name__ == "__main__":
    # preprocess training data
    (x_train, y_train), (x_test, y_test) = pickle.load(open("dataset.pkl", "rb"))
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    with tf.Graph().as_default():
        sess = tf.Session()
        tf.keras.backend.set_session(sess)
        model = ResNet50()

        # model = tf.keras.models.load_model("checkpoint.h5")
        model.summary()

        def top_5_categorical_accuracy(y_true, y_pred):
            return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            metrics=[top_5_categorical_accuracy]
        )

        tb_cb = XTensorboard(log_dir="tensorboard")
        ck_cb = tf.keras.callbacks.ModelCheckpoint("checkpoint.h5", save_best_only=True)
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
                            epochs=100,
                            verbose=1,
                            workers=4,
                            callbacks=[tb_cb, ck_cb, lr_cb])