import os
import numpy as np
import random
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

data_path = "Data/DeepGlobe Land Cover Dataset"

sats_64 = np.load('Data/DeepGlobe Land Cover Dataset/Numpy Arrays/64x64 sat tiles.npy')
masks_64 = np.load('Data/DeepGlobe Land Cover Dataset/Numpy Arrays/64x64 mask tiles.npy')
oh_encoded_64 = np.load('Data/DeepGlobe Land Cover Dataset/Numpy Arrays/64x64 one-hot encoded tiles.npy')

sats_128 = np.load('Data/DeepGlobe Land Cover Dataset/Numpy Arrays/128x128 sat tiles.npy')
masks_128 = np.load('Data/DeepGlobe Land Cover Dataset/Numpy Arrays/128x128 mask tiles.npy')
oh_encoded_128 = np.load('Data/DeepGlobe Land Cover Dataset/Numpy Arrays/128x128 one-hot encoded tiles.npy')



#####
# Get class info
#####

class_df = pd.read_csv(os.path.join(data_path, 'class_dict.csv'))


def extract_tiles(data_path, n_tiles, im_dim=2448, tile_dim=64, fnr=True):
    """Generates numpy arrays to hold training data. The output `sats' contains the satellite tiles, and `masks'
    contains the corresponding pixel labels.

    Args:
        data_path (str): the location of the training data; imagery and masks should be together,
        n_tiles (int): the number of tiles to extract,
        im_dim (int): height of the input image to extract a tile from (input is assumed to be square),
        tile_dim (int): height of the tile to extract (tiles will be square),
        fnr (Boolean): perform a random flip and rotation of each tile,


    Note that this script does not exclude the possibility that two tiles overlaps, or even that the same tile could be
    selected twice. In the future, I would like to ensure that there are no overlaps, but that would take a lot more
    code.
    """

    # generate a list of all file names in the training data, and subset to only the satellite files
    file_names = os.listdir(data_path)
    file_names = [x for x in file_names if x.endswith('jpg')]

    # choose random images (with replacement) from which to extract tiles from
    random.seed(10)
    choices = random.choices(file_names, k=n_tiles)
    sats = np.zeros((0, tile_dim, tile_dim, 3))
    masks = np.zeros((0, tile_dim, tile_dim, 3))

    counter = 0

    for item in choices:

        num = item.partition('_')[0]
        sat_name = num + '_sat.jpg'
        mask_name = num + '_mask.png'

        i = random.randint(0, (im_dim - tile_dim))
        j = random.randint(0, (im_dim - tile_dim))

        if fnr:
            f = random.randint(0, 2)
            r = random.randint(0, 3)

        # Opens the full satellite image, performs a random flip and rotation, then extracts a tile
        with Image.open(os.path.join(data_path, sat_name)) as im:
            if f == 1:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if f == 2:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
            im = im.rotate(r * 90)

            sat_arr = np.asarray(im, dtype=np.uint8)
            sat_tile = sat_arr[i:(i + tile_dim), j:(j + tile_dim)]
            sat_tile = sat_tile.reshape([1, tile_dim, tile_dim, 3])

        # Opens the full mask image, performs a random flip and rotation, then extracts a tile
        with Image.open(os.path.join(data_path, mask_name)) as im:
            if f == 1:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if f == 2:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
            im = im.rotate(r * 90)
            mask_arr = np.asarray(im, dtype=np.uint8)
            mask_tile = mask_arr[i:(i + tile_dim), j:(j + tile_dim)]
            mask_tile = mask_tile.reshape([1, tile_dim, tile_dim, 3])

        sats = np.append(sats, sat_tile, axis=0)
        masks = np.append(masks, mask_tile, axis=0)

        counter +=1
        if counter % 50 == 0:
            print(counter)

    sats /= 255.
    masks = masks.astype(np.uint8)

    return [sats, masks]


def save_tiles(sats, masks, sat_path, mask_path):
    """This function saves the tiled images extracted by extract_tiles() to sat/mask folders of your choice, as png
    files."""

    for i in range(len(sats)):
        sat_arr = sats[i]
        mask_arr = masks[i]

        sat_arr *= 255
        mask_arr *= 255

        sat_arr = sat_arr.astype(np.uint8)
        mask_arr = mask_arr.astype(np.uint8)

        sat_im = Image.fromarray(sat_arr)
        mask_im = Image.fromarray(mask_arr)

        sat_im.save(sat_path + '/' + str(i) + '.png')
        mask_im.save(mask_path + '/' + str(i) + '.png')


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




def view_tiles(sats, masks, predictions, seed=0, num=5):
    """This function outputs a PNG comparing satellite images, their associated ground-truth masks, and a given model's
    prediction. Note that the images are selected randomly from the sats array.

    Args:
        sats (ndarray): a collection of satellite images with shape (#tiles, height, width, 3),
        masks (ndarray): the associated collection of ground-truth masks,
        model (tf.keras.model): the trained model used to make a prediction,
        seed (int): if you want reproducibility, enter this here number,
        num (int): the number of samples to show."""

    # Fixing random state for reproducibility
    if seed:
        np.random.seed(seed)

    choices = np.random.randint(0, len(sats), num)
    newshape = np.insert(sats[0].shape, 0, 1)

    fig, axs = plt.subplots(num, 3)

    for a in range(num):
        if a == 0:
            axs[a, 0].imshow(sats[choices[a]])
            axs[a, 0].set_title("Satellite")
            axs[a, 1].imshow(masks[choices[a]])
            axs[a, 1].set_title("Ground Truth")
            axs[a, 2].imshow(predictions[a])
            axs[a, 2].set_title("Prediction")
        else:
            axs[a, 0].imshow(sats[choices[a]])
            axs[a, 1].imshow(masks[choices[a]])
            axs[a, 2].imshow(predictions[choices[a]])

    plt.setp(axs, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()