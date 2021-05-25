import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dropout, Input, Lambda, \
    MaxPooling2D, UpSampling2D

# ----------------------------------------------------
# UNet
# ----------------------------------------------------

def main_block(m, n_filters, dim, bn, do_rate):
    """The primary convolutional block in the UNet network.
        Args:

            m (tensorflow layer): this is the previous layer in the network,
            n_filters (int): number of filters in each convolution,
            dim (int): the dimension of the filters,
            bn (Boolean): whether to include batch normalization after each convolution,
            do_rate (float): the rate to perform dropout before each convolution."""

    n = Dropout(do_rate)(m)
    n = Conv2D(n_filters, dim, activation='relu', padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do_rate)(n) if do_rate else n
    n = Conv2D(n_filters, dim, activation='relu', padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return n


def unet(im_dim, n_filters, dim=3, do_rate=0, bn=True, opt='Adam', loss='categorical_crossentropy'):
    """Implements the UNet architecture, with options for Dropout and BatchNormalization

        Args:
            im_dim (int): this is the height or width of the inputs tiles (which should be square),
            n_filters (int): the number of filters for the initial layer to have (which doubles each subsequent layer),
            dim (int): the dimension of the filters,
            do_rate (0 <= float <= 1): the rate to perform dropout before each convolution,
            bn (Boolean): whether to include batch normalization after each convolution,
            opt (str): the optimizer for the model (see Keras documentation for options),
            loss (str): the loss function for the model (should only change if creating a custom loss)."""

    # Downsampling path
    input_img = Input(shape=(im_dim, im_dim, 3))

    b1 = main_block(input_img, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    max_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(b1)
    n_filters *= 2
    b2 = main_block(max_1, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    max_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(b2)
    n_filters *= 2
    b3 = main_block(max_2, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    max_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(b3)
    n_filters *= 2
    b4 = main_block(max_3, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)

    # Upsampling path
    up_1 = UpSampling2D(size=(2, 2))(b4)
    con_1 = Concatenate(axis=-1)([up_1, b3])
    n_filters = int(n_filters / 2)
    b5 = main_block(con_1, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    up_2 = UpSampling2D(size=(2, 2))(b5)
    con_2 = Concatenate(axis=-1)([up_2, b2])
    n_filters = int(n_filters / 2)
    b6 = main_block(con_2, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    up_3 = UpSampling2D(size=(2, 2))(b6)
    con_3 = Concatenate(axis=-1)([up_3, b1])
    n_filters = int(n_filters / 2)
    b7 = main_block(con_3, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    output_img = Conv2D(3, 1, padding='same', activation='softmax')(b7)

    # instantiates a model and compiles with the optimizer and loss function
    cnn = Model(input_img, output_img)
    cnn.compile(optimizer=opt, loss=loss)

    return cnn


# ----------------------------------------------------
# DeepWaterMask
# ----------------------------------------------------

class DeepWaterMap:
    """Implements the binary water-detection CNN, with code and data given here:
    https://github.com/isikdogan/deepwatermap. The original implementation put BN layers before the ReLU activation;
    this order is now believed to perform worse. I have updated the architecture to include ReLU before BN."""

    def __init__(self, im_dim):
        self.im_dim = im_dim

    def model(self, min_width = 4, optimizer='Adam', loss='binary crossentropy'):
        inputs = Input(shape=[None, None, 6])

        def conv_block(x, num_filters, kernel_size, stride=1):
            x = Conv2D(
                filters=num_filters,
                kernel_size=kernel_size,
                kernel_initializer='he_uniform',
                strides=stride,
                padding='same',
                activation='relu',
                use_bias=False)(x)
            x = BatchNormalization()(x)

            return x
        def downscaling_unit(x):
            num_filters = int(x.get_shape()[-1]) * 4
            x_1 = conv_block(x, num_filters, kernel_size=5, stride=2)
            x_2 = conv_block(x_1, num_filters, kernel_size=3, stride=1)
            x = Add()([x_1, x_2])

            return x

        def upscaling_unit(x):
            num_filters = int(x.get_shape()[-1]) // 4
            # is the following lambda layer better or worse than UpSampling2D?
            x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
            x_1 = conv_block(x, num_filters, kernel_size=3)
            x_2 = conv_block(x_1, num_filters, kernel_size=3)
            x = Add()([x_1, x_2])

            return x

        def bottleneck_unit(x):
            num_filters = int(x.get_shape()[-1])
            x_1 = conv_block(x, num_filters, kernel_size=3)
            x_2 = conv_block(x_1, num_filters, kernel_size=3)
            x = Add()([x_1, x_2])

            return x

        # model flow
        skip_connections = []
        num_filters = min_width

        # first layer
        x = conv_block(inputs, num_filters, kernel_size=1)
        skip_connections.append(x)

        # encoder
        for i in range(4):
            x = downscaling_unit(x)
            skip_connections.append(x)

        # bottleneck layer
        x = bottleneck_unit(x)

        # decoder
        for i in range(4):
            # do they really want to add the layers, instead of a concatenation?
            x = Add()([x, skip_connections.pop()])
            x = upscaling_unit(x)

        # last layer
        x = Add()([x, skip_connections.pop()])
        x = conv_block(x, 1, kernel_size=1)
        x = Activation('sigmoid')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizer, loss=loss)

        return model



