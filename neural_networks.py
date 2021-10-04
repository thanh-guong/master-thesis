import keras
import keras.models as models
from keras.layers import concatenate
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


def rml201610a_VTCNN2_v2(input_shape, dropout_rate=0.5, classes=11):
    model = models.Sequential()
    model.add(Reshape([1] + input_shape, input_shape=input_shape))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(256, 1, 3, padding="same", activation="relu", name="conv1"))
    model.add(Dropout(dropout_rate))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(80, 2, 3, padding="same", activation="relu", name="conv2"))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', name="dense1"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(classes, name="dense2"))
    model.add(Activation('softmax'))
    model.add(Reshape([classes]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def freehand_v1(input_shape, conv_1_filters=16, conv_1_kernel_size=2, conv_2_filters=16, conv_2_kernel_size=2,
                first_dense_units=256, second_dense_units=256, third_dense_units=128, activation_function="relu",
                classes=11):
    input = keras.Input(shape=input_shape)
    reshape = Reshape(input_shape + [1])(input)
    conv_1 = Convolution2D(conv_1_filters, conv_1_kernel_size, padding="same", activation=activation_function)(reshape)
    fc1 = Dense(first_dense_units, activation=activation_function)(conv_1)
    conv_2 = Convolution2D(conv_2_filters, conv_2_kernel_size, padding="same", activation=activation_function)(fc1)
    max_pool = MaxPooling2D(padding='same')(conv_2)

    out_flatten = Flatten()(max_pool)

    fc2 = Dense(second_dense_units, activation=activation_function)(out_flatten)
    fc3 = Dense(third_dense_units, activation=activation_function)(fc2)
    output = Dense(classes, name="output", activation="softmax")(fc3)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def freehand_v2(input_shape, conv_1_filters=16, conv_1_kernel_size=4, conv_2_filters=32, conv_2_kernel_size=2,
                       first_dense_units=256, second_dense_units=256, third_dense_units=128, activation_function="relu",
                       classes=11):
    # freehand_v2 is freehand_v1 but conv_1_kernel_size is 4 instead of 2 and conv_2_filters are 32 instead of 16
    return freehand_v1(input_shape, conv_1_filters, conv_1_kernel_size, conv_2_filters, conv_2_kernel_size,
                       first_dense_units, second_dense_units, third_dense_units, activation_function, classes)


def freehand_v2_1_double_input(A_input_shape, B_input_shape, conv_1_filters=16, conv_1_kernel_size=4, conv_2_filters=32,
                               conv_2_kernel_size=2, conv_3_filters=16, conv_3_kernel_size=2, first_dense_units=256,
                               second_dense_units=256, third_dense_units=128, activation_function="relu", classes=11):
    # input A
    A_input = keras.Input(shape=A_input_shape)
    A_reshape = Reshape(A_input_shape + [1])(A_input)

    A_conv_1 = Convolution2D(conv_1_filters, conv_1_kernel_size, padding="same", activation=activation_function)(A_reshape)
    A_fc1 = Dense(first_dense_units, activation=activation_function)(A_conv_1)
    A_conv_2 = Convolution2D(conv_2_filters, conv_2_kernel_size, padding="same", activation=activation_function)(A_fc1)
    A_max_pool = MaxPooling2D(padding='same')(A_conv_2)

    # input B
    B_input = keras.Input(shape=B_input_shape)
    B_reshape = Reshape(B_input_shape + [1])(B_input)

    B_conv_1 = Convolution2D(conv_1_filters, conv_1_kernel_size, padding="same", activation=activation_function)(B_reshape)
    B_fc1 = Dense(first_dense_units, activation=activation_function)(B_conv_1)
    B_conv_2 = Convolution2D(conv_2_filters, conv_2_kernel_size, padding="same", activation=activation_function)(B_fc1)
    B_max_pool = MaxPooling2D(padding='same')(B_conv_2)

    # A and B chains concatenation
    x = concatenate([A_max_pool, B_max_pool])

    conv = Convolution1D(conv_3_filters, conv_3_kernel_size, padding="same", activation=activation_function)(x)
    out_flatten = Flatten()(conv)
    fc2 = Dense(second_dense_units, activation=activation_function)(out_flatten)
    fc3 = Dense(third_dense_units, activation=activation_function)(fc2)
    output = Dense(classes, name="output", activation="softmax")(fc3)

    model = keras.Model(inputs=[A_input, B_input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def freehand_v3(input_shape, dropout_rate=0.5, conv_1_filters=16, conv_1_kernel_size=4, conv_2_filters=32,
                conv_2_kernel_size=2, first_dense_units=256, second_dense_units=256, third_dense_units=128,
                activation_function="relu", classes=11):
    input = keras.Input(shape=input_shape)
    reshape = Reshape(input_shape + [1])(input)
    batch_normalization = BatchNormalization()(reshape)

    conv_1 = Convolution2D(conv_1_filters, conv_1_kernel_size, padding="same", activation=activation_function)(batch_normalization)
    batch_normalization_2 = BatchNormalization()(conv_1)
    fc1 = Dense(first_dense_units, activation=activation_function)(batch_normalization_2)
    conv_2 = Convolution2D(conv_2_filters, conv_2_kernel_size, padding="same", activation=activation_function)(fc1)
    batch_normalization_3 = BatchNormalization()(conv_2)
    max_pool = MaxPooling2D(padding='same')(batch_normalization_3)

    out_flatten = Flatten()(max_pool)
    dr = Dropout(dropout_rate)(out_flatten)
    fc2 = Dense(second_dense_units, activation=activation_function)(dr)
    batch_normalization_4 = BatchNormalization()(fc2)
    fc3 = Dense(third_dense_units, activation=activation_function)(batch_normalization_4)
    output = Dense(classes, name="output", activation="softmax")(fc3)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def freehand_v4(input_shape, dropout_rate=0.5, conv_1_filters=16, conv_1_kernel_size=4, conv_2_filters=32,
                conv_2_kernel_size=2, first_dense_units=256, second_dense_units=256, third_dense_units=128,
                activation_function="relu", classes=11):
    input = keras.Input(shape=input_shape)
    reshape = Reshape(input_shape + [1])(input)
    batch_normalization = BatchNormalization()(reshape)

    conv_1 = Convolution2D(conv_1_filters, conv_1_kernel_size, padding="same", activation=activation_function)(
        batch_normalization)
    max_pool = MaxPooling2D(padding='same')(conv_1)
    batch_normalization_2 = BatchNormalization()(max_pool)
    fc1 = Dense(first_dense_units, activation=activation_function)(batch_normalization_2)
    conv_2 = Convolution2D(conv_2_filters, conv_2_kernel_size, padding="same", activation=activation_function)(fc1)
    batch_normalization_3 = BatchNormalization()(conv_2)
    max_pool = MaxPooling2D(padding='same')(batch_normalization_3)

    out_flatten = Flatten()(max_pool)
    dr = Dropout(dropout_rate)(out_flatten)
    fc2 = Dense(second_dense_units, activation=activation_function)(dr)
    batch_normalization_4 = BatchNormalization()(fc2)
    fc3 = Dense(third_dense_units, activation=activation_function)(batch_normalization_4)
    output = Dense(classes, name="output", activation="softmax")(fc3)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def custom_deepsig(input_shape, dropout_rate=0.5, classes=11):

    def conv(filters, kernel_size, previous):
        return Convolution1D(filters, kernel_size, input_shape=input_shape, padding="same", activation="relu")(previous)

    def dropout(previous):
        return Dropout(dropout_rate)(previous)

    def max_pool(previous):
        return MaxPooling1D(padding='same')(previous)

    input = keras.Input(shape=input_shape)

    conv1 = conv(128, 7, input)
    dr_1 = dropout(conv1)
    conv2 = conv(128, 5, dr_1)
    max_pool_1 = max_pool(conv2)

    conv3 = conv(128, 7, max_pool_1)
    dr_2 = dropout(conv3)
    conv4 = conv(128, 5, dr_2)
    max_pool_2 = max_pool(conv4)

    conv5 = conv(128, 7, max_pool_2)
    dr_3 = dropout(conv5)
    conv6 = conv(128, 5, dr_3)
    max_pool_3 = max_pool(conv6)

    conv7 = conv(128, 7, max_pool_3)
    dr_4 = dropout(conv7)
    conv8 = conv(128, 5, dr_4)
    max_pool_4 = max_pool(conv8)

    conv9 = conv(128, 7, max_pool_4)
    dr_5 = dropout(conv9)
    conv10 = conv(128, 5, dr_5)
    max_pool_5 = max_pool(conv10)

    fc1 = Dense(256, name="fc1")(max_pool_5)
    dr_3 = dropout(fc1)
    fc2 = Dense(128, name="fc2")(dr_3)
    out_flatten = Flatten()(fc2)
    output = Dense(classes, name="output")(out_flatten)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def custom_deepsig_v2(input_shape, classes=11):
    iq_in = keras.Input(shape=input_shape, name="IQ")

    conv_1 = Convolution1D(36, 2, padding="same", activation="relu")(iq_in)
    fc1 = Dense(256, activation="relu")(conv_1)
    conv_2 = Convolution1D(18, 4, padding="same", activation="relu")(fc1)
    max_pool = MaxPooling1D(padding='same')(conv_2)

    out_flatten = Flatten()(max_pool)

    fc2 = Dense(256, activation="relu")(out_flatten)
    fc3 = Dense(128, activation="relu")(fc2)
    output = Dense(classes, name="output", activation="softmax")(fc3)

    model = keras.Model(inputs=[iq_in], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def new(input_shape, classes=11, activation_function="relu"):
    inp = keras.Input(shape=input_shape)
    reshape = Reshape(input_shape + [1])(inp)

    conv_1 = Convolution2D(32, kernel_size=(2, 3), padding='same', activation=activation_function)(reshape)
    conv_2 = Convolution2D(64, kernel_size=(2, 3), padding='same', activation=activation_function)(conv_1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv_2)
    dr1 = Dropout(0.25)(max_pool)

    flatten = Flatten()(dr1)

    fc1 = Dense(128, activation=activation_function)(flatten)
    dr2 = Dropout(0.5)(fc1)
    output = Dense(classes, activation="softmax")(dr2)

    model = keras.Model(inputs=[inp], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
