import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dropout, Input, Lambda, \
    MaxPooling2D, UpSampling2D

# ----------------------------------------------------
# Datasets
# ----------------------------------------------------

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

class Potsdam():
    """ISPRS Potsdam semantic segmentation datasets, including Potsdam and Vaihingen separately or combined."""

    def __init__(self, bin_class):
        self.loc = 'This is Potsdam imagery.'
        self.data_path = 'Data/ISPRS Potsdam/Numpy Arrays/'
        if bin_class:
            self.binary = 'This is binary classification imagery for the class ' + '\'' + bin_class + '\'.'
            self.bin_class = bin_class

    def load(self, dim, masks=False, ir=False):
        self.dim = dim

        if ir:
            s = np.load(self.data_path + 'RGBIR_tiles_' + str(dim) + '.npy')
            s = s.astype(np.float32)
            s /= 255
        else:
            s = np.load(self.data_path + 'RGB_tiles_' + str(dim) + '.npy')
            s = s.astype(np.float32)
            s /= 255

        if self.bin_class:
            enc = np.load(self.data_path + 'Binary Classification/' + str(self.bin_class) + '_' + str(self.dim) + '.npy')
        else:
            enc = np.load(self.data_path + 'Encoded_tiles_' + str(dim) + '.npy')

        if masks:
            m = np.load(self.data_path + 'Label_tiles_' + str(dim) + '.npy')
            return [s, m, enc]
        else:
            return [s, enc]

class SemSeg:
    """Loads various data, compiles and fits NN models with options for pretraining, and functions to view results."""

    # initialization properties
    def __init__(self, dim, ir=False, binary_class=None):
        self.dim = dim
        self.ir = ir
        self.data_path = 'Data/ISPRS Potsdam/Numpy Arrays/'
        self.class_df = pd.read_csv(self.data_path + 'class_df.csv')

        if binary_class:
            self.classes = [binary_class]
        else:
            self.classes = (self.class_df)['name'].tolist()

        if ir:
            if binary_class:
                self.summary = 'Class for ' + str(dim) + 'x' + str(dim) + ' binary ' + binary_class + ' imagery, ' + \
                               'with infrared channel.'
            else:
                self.summary = 'Class for ' + str(dim) + 'x' + str(dim) + ' binary ' + binary_class + ' imagery.'
        else:
            self.summary = 'Class for ' + str(dim) + 'x' + str(dim) + ' imagery'

        # the following method is filled from https://keras.io/api/applications/ and may be updated in the future
        self.pretrained_models =['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2',
                                 'ResNet101V2', 'ResNet152V2', 'InceptionV3', 'InceptionResNetV2', 'MobileNet',
                                 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile',
                                 'NASNetLarge', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                                 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

    # if you want separate objects for the data, call this function
    def load_data(self, masks=True, val_split=None):
        if self.ir:
            s = np.load(self.data_path + 'RGBIR_tiles_' + str(self.dim) + '.npy')
        else:
            s = np.load(self.data_path + 'RGB_tiles_' + str(self.dim) + '.npy')

        if len(self.classes) == 1:
            enc = np.load(self.data_path + 'Binary Classification/' + str(self.classes[0]) + '_' + str(self.dim) + '.npy')
        else:
            enc = np.load(self.data_path + 'Encoded_tiles_' + str(self.dim) + '.npy')

        if masks:
            m = np.load(self.data_path + 'Label_tiles_' + str(self.dim) + '.npy')

        if val_split:
            val_size = int(val_split * len(s))



#####
# Helper functions
#####

def rgb_to_binary(rgb_array, class_df, name):
    """This function performs a one-hot encoding of the mask numpy array. Output will have the shape
    (#tiles, height, width, 2). This is only the inverse of binary_to_rgb if the rgb_array is already a binary
    classification.

    Args:
        rgb_array (ndarray): an array of all mask imagery, with shape (#tiles, height, width, depth)
        class_df (data frame): DataFrame with class labels and RGB values in columns (see class_dict.csv above),
        name (str): the name of class_df for the imagery to differentiate.
        """

    n_classes = len(class_df)
    class_ind = class_df[class_df['name'] == name].index[0]

    rgb_list = class_df[['r', 'g', 'b']].values.tolist()
    rgb_list_of_tuples = []

    identity = np.identity(2, dtype=np.uint8)
    one_hot_list = []

    n_tiles = rgb_array.shape[0]
    tile_height = rgb_array.shape[1]
    tile_width = rgb_array.shape[2]

    oh_array = np.zeros((n_tiles, tile_height, tile_width, 2))

    for rgb in rgb_list:
        rgb_tuple = tuple(rgb)
        rgb_list_of_tuples.append(rgb_tuple)

    for row in range(n_classes):
        if row != class_ind:
            one_hot_list.append(tuple(identity[0]))
        else:
            one_hot_list.append(tuple(identity[1]))

    rgb_oh_dict = dict(zip(rgb_list_of_tuples, one_hot_list))

    for s in range(n_tiles):
        # if s % 100 == 0:
        #     print(str(s) + ' complete out of ' + str(n_tiles))
        for h in range(tile_height):
            for w in range(tile_width):
                oh_array[s, h, w] = np.array(rgb_oh_dict[tuple(rgb_array[s, h, w])])

    oh_array = oh_array.astype(np.uint8)
    return oh_array

def binary_to_bw(bin_array):
    """This function takes the one-hot encoded array created by rgb_to_binary(), and returns an RGB array. Output will have
        the shape (#tiles, height, width, 3). This is not the inverse of rgb_to_binary, because this function returns
        only two RGB tuples: black (0, 0, 0) and white (255, 255, 255). Hence, the two functions are only inverses if
        the mask imagery is binary to begin with.

        Args:
            oh_array (ndarray): an array of the one-hot encoded imagery, with shape (#tiles, height, width, #classes).
            """

    rgb_list = [(0, 0, 0), (255, 255, 255)]
    one_hot_list = [(1, 0), (0, 1)]
    oh_rgb_dict = dict(zip(one_hot_list, rgb_list))

    n_tiles, tile_height, tile_width = [bin_array.shape[0], bin_array.shape[1], bin_array.shape[2]]
    rgb_array = np.zeros((n_tiles, tile_height, tile_width, 3))

    for s in range(n_tiles):
        # if s % 100 == 0:
        #     print(str(s) + ' tiles complete out of ' + str(n_tiles))
        for h in range(tile_height):
            for w in range(tile_width):
                rgb_array[s, h, w] = np.array(oh_rgb_dict[tuple(bin_array[s, h, w])])

    rgb_array = rgb_array.astype(np.uint8)
    return rgb_array

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

def pt_model(backbone, input_shape, preprocessing=True, concatenate=True, opt='Adam', loss='categorical_crossentropy'):
    """Instantiates compiled tf.keras.model, with an autoencoder (Unet-like) architecture. The downsampling path is
    given by the 'backbone' argument, with the upsampling path mirroring it, but with options for batch normalization
    and dropout layers.

    Args:
        backbone (str): provides the pretrained model to use for the downsampling path. Must be one of 'Xception',
                        'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', ResNet101V2',
                        'ResNet152V2', 'InceptionV3', 'InceptionResNetV2', 'MobileNet',
                        'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                        'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'. These are the names
                        given at https://keras.io/api/applications/, which may be updated in the future.
        input_shape (tuple): the (height, width, depth) values for the input imagery.
        preprocessing (Boolean): whether to include a preprocessing layer that divides the input by 255.
        concatenate (Boolean): whether to concatenate max-pooling layers from the downsampling to the first tensor of
                               the same shape in the upsampling path.
       opt (str): the optimizer to use when compiling the model.
       loss (str): the loss function to use when compiling the model."""

    x =


def view_tiles(sats, masks, model_a, a_name, model_b, b_name, class_df, bin_class, binary=False, num=5):
    """This function outputs a PNG comparing satellite images, their associated ground-truth masks, and a given model's
    prediction. Note that the images are selected randomly from the sats array.

    Args:
        sats (ndarray): a collection of satellite images with shape (#tiles, height, width, 3),
        masks (ndarray): the associated collection of ground-truth masks,
        pred_a (tf.Keras Model): first model to view images for,
        pred_b (tf.Keras Model): second model to view images for,
        binary (Boolean): if true, masks will be converted to a binary mask,
        class_df (pd.DataFrame): the dataframe containing RGB values for mask,
        bin_class (string): the name from class_df of the class to keep if binary=True,
        num (int): the number of samples to show."""

    choices = np.random.randint(0, len(sats), num)

    s_choices = sats[choices]

    if binary:
        m_choices = binary_to_bw(rgb_to_binary(masks[choices], class_df, name=bin_class))
    else:
        m_choices = masks[choices]

    pred_a = binary_to_bw(vec_to_oh(model_a.predict(sats[choices])))
    pred_b = binary_to_bw(vec_to_oh(model_b.predict(sats[choices])))

    fig, axs = plt.subplots(num, 4)

    for i in range(num):
        if i == 0:
            axs[i, 0].imshow(s_choices[i])
            axs[i, 0].set_title("Satellite")
            axs[i, 1].imshow(m_choices[i])
            axs[i, 1].set_title("Ground Truth")
            axs[i, 2].imshow(pred_a[i])
            axs[i, 2].set_title(str(a_name))
            axs[i, 3].imshow(pred_b[i])
            axs[i, 3].set_title(str(b_name))
        else:
            axs[i, 0].imshow(s_choices[i])
            axs[i, 1].imshow(m_choices[i])
            axs[i, 2].imshow(pred_a[i])
            axs[i, 3].imshow(pred_b[i])

    plt.setp(axs, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------
# Models
# ----------------------------------------------------

def unet_main_block(m, n_filters, dim, bn, do_rate):
    """The primary convolutional block in the UNet network.
        Args:

            m (tensorflow layer): this is the previous layer in the network,
            n_filters (int): number of filters in each convolution,
            dim (int): the dimension of the filters,
            bn (Boolean): whether to include batch normalization after each convolution,
            do_rate (float): the rate to perform dropout before each convolution."""

    n = Conv2D(n_filters, dim, activation='relu', padding='same')(m)
    n = Activation('relu')(n)
    n = Dropout(do_rate)(n) if do_rate else n
    n = BatchNormalization()(n) if bn else n
    n = Conv2D(n_filters, dim, activation='relu', padding='same')(n)
    n = Activation('relu')(n)
    n = Dropout(do_rate)(n) if do_rate else n
    n = BatchNormalization()(n) if bn else n
    return n

#####
# Unet
#####

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

        layer_dict = {}

        # downsampling path
        x = Input(shape=im_dim)
        layer_dict['input'] = x

        for level in range(levels - 1):
            x = unet_main_block(x, n_filters=n_filters, dim=filter_dim_list[level], bn=bn, do_rate=do_rate)
            layer_dict['level_' + str(level)] = x
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
            n_filters *= 2

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


#####
# VGG 16 pretrained feature extraction
#####

model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
model.trainable = False

x = model.output
x = UpSampling2D(size=(2, 2))(x)
# x = Concatenate(axis=-1)([x, model.layers[-5].output])
filters = int(model.output.shape[-1] / 2)
x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=0.2)
x = UpSampling2D(size=(2, 2))(x)
# x = Concatenate(axis=-1)([x, model.layers[-9].output])
filters = int(filters / 2)
x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=0.2)
x = UpSampling2D(size=(2, 2))(x)
# x = Concatenate(axis=-1)([x, model.layers[-13].output])
filters = int(filters / 2)
x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=0.2)
x = UpSampling2D(size=(2, 2))(x)
# x = Concatenate(axis=-1)([x, model.layers[-16].output])
filters = int(filters / 2)
x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=0.2)
x = UpSampling2D(size=(2, 2))(x)

output_img = Conv2D(2, 1, padding='same', activation='softmax')(x)

cnn_pt_no_cat = Model(model.input, output_img)
cnn_pt_no_cat.compile(optimizer='Adam', loss='binary_crossentropy')

