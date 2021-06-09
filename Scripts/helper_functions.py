import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dropout, Input, Lambda, \
    MaxPooling2D, UpSampling2D


#####
# Dataset Classes
#####

class DigitalGlobeDataset():
    """DeepGlobe Land Cover Classification Challenge dataset. Reads in Numpy arrays, converts the satellite image values
     to floats, and provides the land-cover classifications in a dataframe."""

    def __init__(self, data_path):
        self.data_path = data_path

    def class_dict(self):
        class_dict = pd.read_csv(os.path.join(self.data_path, 'class_dict.csv'))

        return class_dict

    def load(self, dim):
        # loads the sat, mask, and one-hot encoded files, while transforming sat values to floats in [0,1]
        assert dim in {64, 128, 256}, "dim parameter must be in {64, 128, 256}"

        sats = np.load(self.data_path + "/Numpy Arrays/" + str(dim) + "x" + str(dim) + " sat tiles.npy").astype(np.float32)
        sats /= 255
        masks = np.load(self.data_path + "/Numpy Arrays/" + str(dim) + "x" + str(dim) + " mask tiles.npy")
        oh_encoded = np.load(self.data_path + "/Numpy Arrays/" + str(dim) + "x" + str(dim) + " one-hot encoded tiles.npy")

        return sats, masks, oh_encoded

class ISPRS():
    """ISPRS semantic segmentation datasets, including Potsdam and Vaihingen separately or combined."""

    def __init__(self):
        self.loc = 'This is Potsdam imagery.'

    def load(self, dim, masks=False, ir=False):
        data_path = 'Data/ISPRS/Numpy Arrays/'

        self.data_path = data_path
        self.dim = dim

        if ir:
            sats = np.load(data_path + 'RGBIR_tiles_' + str(dim) + '.npy')
        else:
            sats = np.load(data_path + 'RGB_tiles_' + str(dim) + '.npy')

        enc = np.load(data_path + 'Encoded_tiles_' + str(dim) + '.npy')

        if masks:
            masks = np.load(data_path + 'Label_tiles_' + str(dim) + '.npy')
            return [sats, masks, enc]
        else:
            return [sats, enc]

#####
# Helper functions
#####

def rgb_to_oh(rgb_array, class_df):
    """This function performs a one-hot encoding of the mask numpy array. Output will have the shape
    (#tiles, height, width, #classes). This is the inverse of oh_to_rgb().

    Args:
        rgb_array (ndarray): an array of all mask imagery, with shape (#tiles, height, width, depth)
        class_df (data frame): DataFrame with class labels and RGB values in columns (see class_dict.csv above)
        """

    n_classes = len(class_df)
    rgb_list = class_df[['r', 'g', 'b']].values.tolist()
    rgb_list_of_tuples = []

    identity = np.identity(n_classes, dtype=np.float32)
    one_hot_list = []

    n_tiles = rgb_array.shape[0]
    tile_height = rgb_array.shape[1]
    tile_width = rgb_array.shape[2]

    oh_array = np.zeros((n_tiles, tile_height, tile_width, n_classes))

    for rgb in rgb_list:
        rgb_tuple = tuple(rgb)
        rgb_list_of_tuples.append(rgb_tuple)

    for row in range(n_classes):
        one_hot_list.append(tuple(identity[row]))

    rgb_oh_dict = dict(zip(rgb_list_of_tuples, one_hot_list))

    for s in range(n_tiles):
        if s % 100 == 0:
            print(str(s) + ' complete out of ' + str(n_tiles))
        for h in range(tile_height):
            for w in range(tile_width):
                oh_array[s, h, w] = np.array(rgb_oh_dict[tuple(rgb_array[s, h, w])])

    oh_array = oh_array.astype(np.uint8)
    return oh_array


def oh_to_rgb(oh_array, class_df):
    """This function takes the one-hot encoded array created by rgb_to_oh(), and returns an RGB array. Output will have
    the shape (#tiles, height, width, 3). This is also the inverse of oh_to_rgb().

    Args:
        oh_array (ndarray): an array of the one-hot encoded imagery, with shape (#tiles, height, width, #classes)
        class_df (data frame): DataFrame with class labels and RGB values in columns (see class_dict.csv above)
        """

    n_classes = len(class_df)
    rgb_list = class_df[['r', 'g', 'b']].values.tolist()
    rgb_list_of_tuples = []

    identity = np.identity(n_classes)
    one_hot_list = []

    n_tiles = oh_array.shape[0]
    tile_height = oh_array.shape[1]
    tile_width = oh_array.shape[2]

    rgb_array = np.zeros((n_tiles, tile_height, tile_width, 3))

    for rgb in rgb_list:
        rgb_tuple = tuple(rgb)
        rgb_list_of_tuples.append(rgb_tuple)

    for row in range(n_classes):
        one_hot_list.append(tuple(identity[row]))

    oh_rgb_dict = dict(zip(one_hot_list, rgb_list_of_tuples))

    for s in range(n_tiles):
        # if s % 100 == 0:
        #     print(str(s) + ' tiles complete out of ' + str(n_tiles))
        for h in range(tile_height):
            for w in range(tile_width):
                rgb_array[s, h, w] = np.array(oh_rgb_dict[tuple(oh_array[s, h, w])])

    rgb_array = rgb_array.astype(np.uint8)
    return rgb_array


def vec_to_oh(array, progress=False, cycle=100):
    """This function takes "array" and converts its depth vectors to one-hot encodings.

    Args:
        array (ndarray): numpy array that is likely the output of a NN model prediction,
        progress (Boolean): if applied to a large array, toggle this to True to get the progress,
        cycle (int): determines how often to print a report."""

    [samples, height, width, depth] = array.shape
    identity = np.identity(depth)

    oh_array = np.zeros(array.shape)

    for i in range(samples):
        for j in range(height):
            for k in range(width):
                hot_spot = np.argmax(array[i, j, k])
                oh_array[i, j, k, hot_spot] = 1

        if progress and (i % cycle == 0):
            print(str(i) + ' tiles complete out of ' + str(samples))

    oh_array = oh_array.astype(np.uint8)
    return oh_array


def view_tiles(sats, masks, preds, seed=0, num=5):
    """This function outputs a PNG comparing satellite images, their associated ground-truth masks, and a given model's
    prediction. Note that the images are selected randomly from the sats array.

    Args:
        sats (ndarray): a collection of satellite images with shape (#tiles, height, width, 3),
        masks (ndarray): the associated collection of ground-truth masks,
        preds (ndarray): the predicted images given by the model,
        seed (int): if you want reproducibility, enter this here number,
        num (int): the number of samples to show."""

    # Fixing random state for reproducibility
    if seed:
        np.random.seed(seed)

    choices = np.random.randint(0, len(sats), num)

    fig, axs = plt.subplots(num, 3)

    for a in range(num):
        if a == 0:
            axs[a, 0].imshow(sats[choices[a]])
            axs[a, 0].set_title("Satellite")
            axs[a, 1].imshow(masks[choices[a]])
            axs[a, 1].set_title("Ground Truth")
            axs[a, 2].imshow(preds[a])
            axs[a, 2].set_title("Prediction")
        else:
            axs[a, 0].imshow(sats[choices[a]])
            axs[a, 1].imshow(masks[choices[a]])
            axs[a, 2].imshow(preds[choices[a]])

    plt.setp(axs, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


def unet_main_block(m, n_filters, dim, bn, do_rate):
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


# ----------------------------------------------------
# Models
# ----------------------------------------------------

class Unet:
    def __init__(self, classes):
        self.classes = classes

    def model(self, im_dim, n_filters, levels, filter_dims, do_rate, bn=True, opt='Adam',
              loss='categorical_crossentropy'):
        """Implements the Unet architecture.

        Args:
            im_dim (tuple): the shape of an input image,
            n_filters (int): the number of filters in the first/last convolutional layers,
            levels (int): the number of levels in the network, including the bottom of the 'U',
            filter_dims (int or list-like): the size of the filters in the respective convolutional layers,
            do_rate (0 <= float <= 1): the dropout rate,
            bn (Boolean): whether to include batch-normalization,
            opt (string): the optimizer to compile the model with,
            loss (string): the loss function to compile the model with."""

        assert (im_dim[0] / (2 ** levels) >= 1.0), 'Too many levels for this input image size.'

        if isinstance(filter_dims, int):
            filter_dim_list = []
            for level in range(levels):
                filter_dim_list.append(filter_dims)
        else:
            filter_dim_list = list(map(int, filter_dims))
            assert (levels == len(filter_dim_list)), 'Specify the same number of filter dimensions as levels.'

        print(filter_dim_list)

        layer_dict = {}

        # downsampling path
        x = Input(shape=im_dim)
        layer_dict['input'] = x

        for level in range(levels - 1):
            x = unet_main_block(x, n_filters=n_filters, dim=filter_dim_list[level], bn=bn, do_rate=do_rate)
            layer_dict['level_' + str(level)] = x
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
            n_filters *= 2

        print(layer_dict)

        # lowest level
        x = unet_main_block(x, n_filters=n_filters, dim=filter_dim_list[-1], bn=bn, do_rate=do_rate)

        # upsampling path
        for level in range(levels - 1):
            x = UpSampling2D(size=(2, 2))(x)
            x = Concatenate(axis=-1)([x, layer_dict['level_' + str(levels - (level + 2))]])
            n_filters = int(n_filters / 2)
            x = unet_main_block(x, n_filters=n_filters, dim=filter_dim_list[1 - level], bn=bn, do_rate=do_rate)

        output_img = Conv2D(self.classes, 1, padding='same', activation='softmax')(x)

        cnn = Model(layer_dict['input'], output_img)
        cnn.compile(optimizer=opt, loss=loss)

        return cnn


unet = Unet(classes=6)
model = unet.model((256, 256, 3), 16, 9, 3, 0.2)

def unet_small(im_dim, n_filters, classes, dim=3, do_rate=0, bn=True, opt='Adam', loss='categorical_crossentropy'):
    """Implements the UNet architecture, with options for Dropout and BatchNormalization

        Args:
            im_dim (int): this is the height or width of the inputs tiles (which should be square),
            n_filters (int): the number of filters for the initial layer to have (which doubles each subsequent layer),
            dim (int): the dimension of the filters,
            do_rate (0<=float<=1): the rate to perform dropout before each convolution,
            bn (Boolean): whether to include batch normalization after each convolution,
            opt (str): the optimizer for the model (see Keras documentation for options),
            loss (str): the loss function for the model (should only change if creating a custom loss)."""

    # Downsampling path
    input_img = Input(shape=(im_dim, im_dim, 3))

    b1 = unet_main_block(input_img, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    max_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(b1)
    n_filters *= 2
    b2 = unet_main_block(max_1, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    max_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(b2)
    n_filters *= 2
    b3 = unet_main_block(max_2, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    max_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(b3)
    n_filters *= 2
    b4 = unet_main_block(max_3, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)

    # Upsampling path
    up_1 = UpSampling2D(size=(2, 2))(b4)
    con_1 = Concatenate(axis=-1)([up_1, b3])
    n_filters = int(n_filters / 2)
    b5 = unet_main_block(con_1, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    up_2 = UpSampling2D(size=(2, 2))(b5)
    con_2 = Concatenate(axis=-1)([up_2, b2])
    n_filters = int(n_filters / 2)
    b6 = unet_main_block(con_2, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    up_3 = UpSampling2D(size=(2, 2))(b6)
    con_3 = Concatenate(axis=-1)([up_3, b1])
    n_filters = int(n_filters / 2)
    b7 = unet_main_block(con_3, n_filters=n_filters, dim=dim, bn=bn, do_rate=do_rate)
    output_img = Conv2D(classes, 1, padding='same', activation='softmax')(b7)

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