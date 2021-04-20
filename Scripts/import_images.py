import os
import numpy as np
import random
import pandas as pd
from PIL import Image
from sklearn import preprocessing

data_path = "Data/DeepGlobe Land Cover Dataset"

sats = np.load('Data/DeepGlobe Land Cover Dataset/10000 DG sat tiles.npy')
masks = np.load('Data/DeepGlobe Land Cover Dataset/10000 DG mask tiles.npy')

#####
# Get class info
#####

class_df = pd.read_csv(os.path.join(data_path, 'class_dict.csv'))
# Get class names
class_names = class_df['name'].tolist()
# Get class RGB values
class_rgb_values = class_df[['r', 'g', 'b']].values.tolist()

#####
# Perform a one-hot encoding of the land-cover classes
scaled_rgb = np.asarray(class_df[['r', 'g', 'b']].values / 255, dtype=np.float32)




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
            sat_tile = sat_arr[i:(i + 64), j:(j + 64)]
            sat_tile = sat_tile.reshape([1, tile_dim, tile_dim, 3])

        # Opens the full mask image, performs a random flip and rotation, then extracts a tile
        with Image.open(os.path.join(data_path, mask_name)) as im:
            if f == 1:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if f == 2:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
            im = im.rotate(r * 90)
            mask_arr = np.asarray(im, dtype=np.uint8)
            mask_tile = mask_arr[i:(i + 64), j:(j + 64)]
            mask_tile = mask_tile.reshape([1, tile_dim, tile_dim, 3])

        sats = np.append(sats, sat_tile, axis=0)
        masks = np.append(masks, mask_tile, axis=0)

        counter +=1
        if counter % 50 == 0:
            print(counter)

    sats /= 255.
    masks /= 255.

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
