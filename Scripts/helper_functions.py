import os
import numpy as np
import random
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

data_path = "Data/DeepGlobe Land Cover Dataset/train"

class DigitalGlobeDataset():
    """DeepGlobe Land Cover Classification Challenge dataset. Reads in Numpy arrays, converts the satellite image values
     to floats, and provides the land-cover classifications in a dataframe."""

    def __init__(self, data_path):
        self.data_path = data_path

    def class_dict(self):
        class_dict = pd.read_csv(os.path.join(data_path, 'class_dict.csv'))

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

    def __init__(self, loc):
        self.loc = loc
        assert loc in {'Potsdam', 'Vaihingen'}, 'Location must be one of \'Potsdam\', \'Vaihingen\'.'

    def load(self, dim, masks = True, ir=False):
        data_path = 'Data/ISPRS/' + self.loc + '/Numpy Arrays/'

        self.data_path = data_path
        self.dim = dim

        if ir:
            sats = np.load(data_path + 'RGBIR_tiles_' + str(dim) + '.npy')
        else:
            sats = np.load(data_path + 'RGB_tiles_' + str(dim) + '.npy')

        enc = np.load(data_path + 'Encoded_tiles_' + str(dim) + '.npy')

        if masks:
            masks = np.load(data_path + 'RGBIR_tiles_' + str(dim) + '.npy')
            return [sats, masks, enc]
        else:
            return [sats, enc]


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
