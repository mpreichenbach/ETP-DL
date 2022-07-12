import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import xception, vgg16, vgg19, resnet, resnet_v2
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D
import time


def mosaics2tiles(rgb, mask, tile_dim, require_labels=(0, 1), drop_nodata_tiles=True,
                 rgb_save_path=None, mask_save_path=None, dtype=np.uint8, verbose=20):
    """Takes a list of image arrays, and generates a Numpy array of shape (n, tile_dim, tile_dim, depth) for each member
    of the list. Note that this will, by default, cut off the right and bottom edges if tile_dim does not divide the
    mosaic dimensions evenly.

    Args:
        mosaic_list (list): a list of ndarrays, we assume [0] is imagery and [1] is the mask;
        tile_dim (int): the size of tiles to generate;
        dtype (Numpy dtype): the datatype for the output arrays;
        drop_nodata_tiles (bool): whether to ignore tiles which have a 0 in each depth dimension;
        verbose (int): the number of generated tiles after which to print a progress report; 0 for no report."""

    # when labels are stored as 'binary', cv2 will import 1 as 255; this corrects for that
    if 255 in np.unique(mask):
        mask = mask / 255
        mask = mask.astype(dtype=dtype)

    require_labels = set(require_labels)
    mosaic_list = [rgb, mask]

    # check that both or neither save paths are specified
    if type(rgb_save_path) != type(mask_save_path):
        raise Exception("Both save path arguments must be either None or strings.")

    # check that relevant dimensions of all arrays match
    shape_list = []
    for arr in [rgb, mask]:
        if len(arr.shape) > 2:
            shape_list.append(arr.shape[0:-1])
        else:
            shape_list.append(arr.shape)

    if len(set(shape_list)) > 1:
        raise Exception("Input arrays do not all have the same dimensions.")

    # create the lists to hold the tiles (assumes there are only two: [imagery, labels])
    imagery_tiles = []
    mask_tiles = []

    imagery_array = mosaic_list[0]
    mask_array = mosaic_list[1]

    height = imagery_array.shape[0]
    width = imagery_array.shape[1]
    for i in range(int(height / tile_dim)):
        for j in range(int(width / tile_dim)):
            imagery_tile = imagery_array[tile_dim * i: tile_dim * (i + 1), tile_dim * j: tile_dim * (j + 1)]
            mask_tile = mask_array[tile_dim * i: tile_dim * (i + 1), tile_dim * j: tile_dim * (j + 1)]

            # We assume nodata pixels are 0 for each depth dimension; this checks whether that's true anywhere in the
            # tile, and whether the labels have only one value; if all this is true, the loop moves to the next tile.
            depth_sum = np.sum(imagery_tile, axis=-1)
            if drop_nodata_tiles and 0 in np.unique(depth_sum) or not require_labels.issubset(np.unique(mask_tile)):
                continue
            else:
                if rgb_save_path is not None:
                    cv2.imwrite(rgb_save_path + "_" + str(len(imagery_tiles)) + ".png", imagery_tile)
                if mask_save_path is not None:
                    cv2.imwrite(mask_save_path + "_" + str(len(mask_tiles)) + ".png", mask_tile)

                # reshape to allow easy concatenate later
                imagery_tile = np.expand_dims(imagery_tile, 0)
                mask_tile = np.expand_dims(mask_tile, 0)

                imagery_tiles.append(imagery_tile)
                mask_tiles.append(mask_tile)

            if verbose and len(imagery_tiles) % verbose == 0:
                n_tiles = len(imagery_tiles)
                max_tiles = int(height * width / tile_dim ** 2)
                print("Out of a maximum of " + str(max_tiles) + ", " + str(n_tiles) + " complete.")

    if rgb_save_path is None and mask_save_path is None:
        rgb_tile_array = np.concatenate(imagery_tiles, axis=0, dtype=dtype)
        label_tile_array = np.concatenate(mask_tiles, axis=0, dtype=dtype)
        return rgb_tile_array, label_tile_array
    else:
        print("Tile saving complete.")


def reduce_classes(array, keep_labels=None):
    """Takes array of labels and keeps only the classes given in keep_labels. If keep_labels=[a, b], then output will
    have values [0, 1, 2], where 1 corresponds to a, 2 corresponds to b, and 0 corresponds to all values in array which
    are neither a nor b.

    Args:
        array (ndarray): an array with shape (n_images, height, width), and integer classes,
        keep_labels (list or integer): either a single integer, or a list of the labels to keep."""

    if isinstance(keep_labels, int):
        vals = [keep_labels]
    elif isinstance(keep_labels, list):
        vals = keep_labels

    # if input array is integer labels (masks)
    if len(array.shape) != 3:
        print("Input array must have shape (n_images, dim, dim).")
        return
    else:
        out_array = np.zeros(array.shape, dtype=np.uint8)
        new_vals = (np.arange(len(vals)) + 1).tolist()
        for i in range(len(vals)):
            val = vals[i]
            new_val = new_vals[i]
            out_array = np.where(array == val, new_val, out_array)

    return out_array.astype(np.uint8)

def pt_model(n_classes, n_filters=None, concatenate=True, do=0.2, opt='Adam', loss='sparse_categorical_crossentropy'):
    """Instantiates compiled tf.keras.model, with an autoencoder (Unet-like) architecture. The downsampling path is
    given by the 'backbone' argument, with the upsampling path mirroring it, but with options for batch normalization
    and dropout layers.

    When saving and loading models generated from this function, use the model.save_weights and model.load_weights
    methods; model.save followed by tf.keras.models.load_model often does not work.

    Args:
        backbone (str): provides the pretrained model to use for the downsampling path. Must be one of 'Xception',
                        'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', ResNet101V2',
                        'ResNet152V2', 'InceptionV3', 'InceptionResNetV2', 'MobileNet',
                        'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                        'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'. These are the names
                        given at https://keras.io/api/applications/, which may be updated in the future.
        input_shape (tuple): the (height, width, depth) values for the input imagery.
        n_classes (int): the number of classes to predict.
        concatenate (Boolean): whether to concatenate max-pooling layers from the downsampling to the first tensor of
                               the same shape in the upsampling path.
       opt (str): the optimizer to use when compiling the model.
       loss (str): the loss function to use when compiling the model."""

    input = Input(shape=(None, None, 3), dtype=tf.float32)

    down_path = vgg19.VGG19(include_top=False, input_tensor=input)
    for i in range(len(down_path.layers)):
        down_path.layers[i].trainable=False

    x = down_path.output

    # upsampling path
    filters = int(down_path.output.shape[-1])
    x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=-1)([x, down_path.layers[-6].output]) if concatenate else x
    filters = int(filters / 2)
    x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=-1)([x, down_path.layers[-11].output]) if concatenate else x
    filters = int(filters / 2)
    x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=-1)([x, down_path.layers[-16].output]) if concatenate else x
    filters = int(filters / 2)
    x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=-1)([x, down_path.layers[-19].output]) if concatenate else x
    filters = int(filters / 2)
    x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
    x = UpSampling2D(size=(2, 2))(x)
    output = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

    # compile the model with the chosen optimizer and loss functions
    cnn_pt = Model(input, output)
    cnn_pt.compile(optimizer=opt, loss=loss)

    return cnn_pt

def vec_to_label(oh_array):
    output = np.argmax(oh_array, axis=-1).astype(np.uint8)

    return output

def label_to_oh(label_array, classes):
    """Converts integer labels with shape (n_images, height, width) to one-hot encodings."""

    initial_dim = len(label_array.shape)

    enc = np.zeros(label_array.shape + (classes,), dtype=np.uint8)

    for i in range(classes):
        if initial_dim == 2:
            enc[:, :, i][label_array == i] = 1
        if initial_dim == 3:
            enc[:, :, :, i][label_array == i] = 1

    return enc

def tile_apply(image, model, tile_dim, output_type=np.uint8):
    """Tiles an image, model predicts each tile, returns labeled image with original dimensions. Later versions of this
    functions should have options for overlap, and continuous extension at the edges. For now, make sure that the image
    dimensions are powers of 2.

    Args:
        image (file object): an image-file that can be input into np.asarray,
        model (Keras model): the model with which to do the predicting,
        tile_dim (0 < large power of 2 < memory limit): the dimensions of square image tiles for the model to predict,
        output_type (numpy data type): the data type for the output array."""

    arr = np.asarray(image)

    holder = np.zeros(arr.shape[0:2])

    nrows = int(arr.shape[0] / tile_dim)
    ncols = int(arr.shape[1] / tile_dim)

    print("Classifying tiles of shape " + str((tile_dim, tile_dim, 3)) +
          " from image with dimensions " + str(arr.shape) + ".")
    tic = time.perf_counter()
    for i in range(nrows):
        for j in range(ncols):
            tile = arr[(tile_dim * i):(tile_dim * (i + 1)), (tile_dim * j):(tile_dim * (j + 1))]
            pred = vec_to_label(model.predict(tile.reshape((1,) + tile.shape))[0])

            holder[(tile_dim * i):(tile_dim * (i + 1)), (tile_dim * j):(tile_dim * (j + 1))] = pred
    toc = time.perf_counter()
    t_elapsed = round(toc - tic, 4)
    print("Classification complete; time elapsed: " + str(t_elapsed) + " seconds.")

    holder = holder.astype(output_type)

    return holder

def vec_to_oh(array):
    """This function takes "array" and converts its depth-wise probability vectors to one-hot encodings.

    Args:
        array (ndarray): any array, but likely the output of a Keras model prediction."""

    comparison = np.equal(np.amax(array, axis=-1, keepdims=True), array)
    oh_array = np.where(comparison, 1, 0).astype(np.uint8)

    return oh_array

def view_tiles(sats, masks, models, n_tiles=5, classes=6, choices=None, cmap='Accent', display=True, path=None):
    """This function outputs a PNG comparing satellite images, their associated ground-truth masks, and a given model's
    prediction. Note that the images are selected randomly from the sats array.

    Args:
        sats (ndarray): a collection of satellite images with shape (#tiles, height, width, 3),
        masks (ndarray): the associated collection of ground-truth masks,
        model_list (list): a list of Keras model objects,
        num (int): the number of tiles to show
        choices (Numpy array): a vector of indices in range(len(sats)) to select specific tiles to display,
        cmap (str): a Matplotlib colorplot name,
        display (Boolean): whether to display the figure,
        path (str or None): where to save the figure.
        """

    n_models = len(models)
    model_list = models

    if choices is not None:
        idx = choices
        n_tiles = len(idx)
    else:
        idx = np.random.choice(len(sats), n_tiles, replace=True)

    s_choices = sats[idx]
    m_choices = masks[idx]
    pred_list = []

    for model in model_list:
        arr = vec_to_label(model.predict(s_choices)).astype(np.uint8)
        pred_list.append(arr)

    # this ensures that the colormapping knows the full range of labels for each imshow call below.
    norm = Normalize(vmin=0, vmax=classes-1)

    fig, axs = plt.subplots(n_tiles, 2 + n_models, constrained_layout=True)

    for i in range(n_tiles):
        if i == 0:
            axs[i, 0].imshow(s_choices[i])
            axs[i, 0].set_title("RGB")
            axs[i, 1].imshow(m_choices[i], cmap=cmap, norm=norm)
            axs[i, 1].set_title("Truth")
            for j in range(n_models):
                axs[i, 2 + j].imshow(pred_list[j][i], cmap=cmap, norm=norm)
                axs[i, 2 + j].set_title(model_list[j].name)

        else:
            axs[i, 0].imshow(s_choices[i])
            axs[i, 1].imshow(m_choices[i], cmap=cmap, norm=norm)
            for j in range(n_models):
                axs[i, 2 + j].imshow(pred_list[j][i], cmap=cmap, norm=norm)

    plt.setp(axs, xticks=[], yticks=[])

    if display:
        plt.show()

    if path is not None:
        plt.savefig(path, bbox_inches='tight')

def unet_main_block(m, n_filters, dim=3, bn=True, do_rate=0.2):
    """The primary convolutional block in the UNet network.

        Args:
            m (Keras layer): this is the previous layer in the network,
            n_filters (int): number of filters in each convolutional layer,
            dim (int): the height/width dimension of the filters),
            bn (Boolean): whether to include batch normalization
            do_rate (float): the rate to perform dropout after each convolution."""

    n = Conv2D(n_filters, dim, activation='relu', padding='same')(m)
    n = Dropout(do_rate)(n) if do_rate else n
    n = BatchNormalization()(n) if bn else n
    n = Conv2D(n_filters, dim, activation='relu', padding='same')(n)
    n = Dropout(do_rate)(n) if do_rate else n
    n = BatchNormalization()(n) if bn else n

    return n
