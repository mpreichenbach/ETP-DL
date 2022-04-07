import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import xception, vgg16, vgg19, resnet, resnet_v2
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dropout, Input, Lambda, UpSampling2D
import time

def iou_loss(mask, pred):
    """Computes the iou_loss for binary input tensors.

    Args:
        mask (numpy array): the array of true labels (should be one-hot encoded),
        pred (numpy array): the array of predicted labels (one-hot encoded)."""

    # convert probability vector to label
    pred = K.argmax(pred, axis=-1)

    intersection = K.prod(mask, pred)
    union = mask + pred - intersection

    iou = intersection / union

    return 1 - iou

def iou_score(mask, pred):
    """See https://en.wikipedia.org/wiki/Jaccard_index; we follow the notation of the first section."""

    if (mask.shape != pred.shape):
        raise Exception("Input arrays have different shapes.")

    intersection = np.multiply(mask, pred)
    union = mask + pred - intersection

    iou = np.sum(intersection) / np.sum(union)

    return iou

def total_acc(mask, pred):
    """Accuracy over all pixels."""

    if (mask.shape != pred.shape):
        raise Exception("Input arrays have different shapes.")

    acc = np.sum(np.where(mask == pred, 1, 0)) / (mask.shape[0] ** 2)

    return acc

def label_acc(mask, pred, label):
    """Accuracy on only the given label."""

    mask_labels = np.unique(mask).tolist()
    pred_labels = np.unique(pred).tolist()

    all_labels = set(mask_labels + pred_labels)

    if (label not in all_labels):
        raise Exception("Label " + str(label) + " is not in at least one of the input arrays.")

    if (mask.shape != pred.shape):
        raise Exception("Input arrays have different shapes.")

    mask_holder = np.where(mask == label, 1, 0)
    pred_holder = np.where(pred == label, 1, 0)

    acc = np.sum(np.multiply(mask_holder, pred_holder)) / np.sum(mask_holder)

    return acc

def rotate(x):
    """Performs a random rotation on the input image."""
    return np.rot90(x, np.random.randint(0, 4))

def numpy2img(array, path, type = "png", progress = 0):
    """Given an array with dimensions (n_tiles, height, width, depth), saves the individual tiles as images of format
    'type', in the folder given by path. The file name will be the corresponding n_tiles index.

    Args:
        array (ndarray): an array with dimensions (n_tiles, height, width, depth),
        path (string): the path in which to save image files,
        type (string): the desired file type,
        progress (int)): gives the number of images to save before printing an update; if 0, no update is printed. """

    if not isinstance(array, np.ndarray):
        print("Argument 'array' must be of class np.ndarray.")
        return
    if not isinstance(path, str):
        print("Argument 'path' must be a string.")
        return
    if not isinstance(type, str):
        print("Argument 'type' must be a string.")
        return

    if path[-1] != "/":
        print("Argument 'path' must end with /.")
        return

    n_images = len(array)

    for i in range(n_images):
        im = Image.fromarray(array[i])
        im.save(fp = path + str(i) + "." + type)
        if (progress != 0) & (i % progress == 0):
            print("Finished saving " + str(i) + "/" + str(n_images) + " images.")


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


def pt_model(backbone, n_classes, concatenate=True, do=0.2, opt='Adam', loss='sparse_categorical_crossentropy'):
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

    # testing the following line:
    input = Input(shape=(None, None, 3), dtype=tf.float32)

    # input = Input(input_shape, dtype=tf.float32)

    #####
    # Xception
    #####
    if backbone == 'Xception':
        input_proc = xception.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = xception.Xception(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path
        filters = int(model_pt.output.shape[-1])
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
        cnn_pt.compile(optimizer=opt, loss=loss)

        return cnn_pt

    #####
    # VGG16 & VGG19
    #####
    if backbone == 'VGG16':
        input_proc = vgg16.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = vgg16.VGG16(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path
        filters = int(model_pt.output.shape[-1])
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-5].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-9].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-13].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-16].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
        cnn_pt.compile(optimizer=opt, loss=loss)

        return cnn_pt

    if backbone == 'VGG19':
        input_proc = vgg19.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = vgg19.VGG19(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path
        filters = int(model_pt.output.shape[-1])
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-6].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-11].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-16].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-19].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output)
        cnn_pt.compile(optimizer=opt, loss=loss)

        return cnn_pt

    #####
    # ResNet50 & ResNet50V2
    #####

    if backbone == 'ResNet50':
        input_proc = resnet.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = resnet.ResNet50(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path

        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-33].output]) if concatenate else x
        # I've started with half the filters as before, because otherwise I get a GPU memory error
        filters = int(model_pt.output.shape[-1] / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-95].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-137].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-171].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
        cnn_pt.compile(optimizer=opt, loss=loss)

        del model_pt

        return cnn_pt

    if backbone == 'ResNet50V2':
        input_proc = resnet_v2.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = resnet_v2.ResNet50V2(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-44].output]) if concatenate else x
        # I've started with half the filters as in model_pt.output, because otherwise I get a GPU memory error
        filters = int(model_pt.output.shape[-1] / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-112].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-158].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-188].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
        cnn_pt.compile(optimizer=opt, loss=loss)

        return cnn_pt

    #####
    # ResNet101 & ResNet101V2
    # During training, use a batch size of 8; 16 gives a GPU memory error
    #####

    if backbone == 'ResNet101':
        input_proc = resnet.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = resnet.ResNet101(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-33].output]) if concatenate else x
        # I've started with half the filters as in model_pt.output, because otherwise I get a GPU memory error
        filters = int(model_pt.output.shape[-1] / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-265].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-307].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-341].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
        cnn_pt.compile(optimizer=opt, loss=loss)

        return cnn_pt

    if backbone == 'ResNet101V2':
        input_proc = resnet_v2.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = resnet_v2.ResNet101V2(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-44].output]) if concatenate else x
        # I've started with half the filters as in model_pt.output, because otherwise I get a GPU memory error
        filters = int(model_pt.output.shape[-1] / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-299].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-345].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-375].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
        cnn_pt.compile(optimizer=opt, loss=loss)

        return cnn_pt

    #####
    # ResNet152 & ResNet152V2
    # During training, use a batch size of 8, 16 gives a GPU memory error
    #####

    if backbone == 'ResNet152':
        input_proc = resnet.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = resnet.ResNet152(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-33].output]) if concatenate else x
        # I've started with 1/4 the filters as in model_pt.output, because otherwise I get a GPU memory error
        filters = int(model_pt.output.shape[-1] / 4)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-395].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-477].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-511].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
        cnn_pt.compile(optimizer=opt, loss=loss)

        return cnn_pt

    if backbone == 'ResNet152V2':
        input_proc = resnet_v2.preprocess_input(input)
        input_model = Model(input, input_proc)
        model_pt = resnet_v2.ResNet152V2(include_top=False, input_tensor=input_model.output)
        model_pt.trainable = False

        x = model_pt.output

        # upsampling path
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-44].output]) if concatenate else x
        # I've started with 1/4 the filters as in model_pt.output, because otherwise I get a GPU memory error
        filters = int(model_pt.output.shape[-1] / 4)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-442].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-532].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, model_pt.layers[-562].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
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

    holder = np.zeros(arr.shape[0:2] + (model.output.shape[-1],))

    nrows = int(arr.shape[0] / tile_dim)
    ncols = int(arr.shape[1] / tile_dim)

    print("Classifying tiles of shape " + str((tile_dim, tile_dim, 3)) +
          " from image with dimensions " + str(arr.shape) + ".")
    tic = time.perf_counter()
    for i in range(nrows):
        for j in range(ncols):
            tile = arr[(tile_dim * i):(tile_dim * (i + 1)), (tile_dim * j):(tile_dim * (j + 1))]
            pred = model.predict(tile.reshape((1,) + tile.shape))[0]

            holder[(tile_dim * i):(tile_dim * (i + 1)), (tile_dim * j):(tile_dim * (j + 1))] = pred
    toc = time.perf_counter()
    t_elapsed = round(toc - tic, 4)
    print("Classification complete; time elapsed: " + str(t_elapsed) + " seconds.")

    holder = holder.astype(output_type)

    return holder

def tvt_split(array, splits=[0.7, 0.2]):
    """Creates a training/validation/test split of an array.

    Args:
        array (array-like): the array of shape (batch size, height, width, depth) to be split,
        splits (list): the proportions to put into training/validation, respectively; the remainder is testing."""

    arr = np.asarray(array)
    train_size = int(splits[0] * arr.shape[0])
    val_size = int(splits[1] * arr.shape[0])

    train_choices = np.random.choice(arr.shape[0], train_size, replace=False)
    training = arr[train_choices]

    holder = np.delete(arr, train_choices, axis=0)
    val_choices = np.random.choice(holder.shape[0], val_size, replace=False)
    validation = holder[val_choices]
    testing = np.delete(holder, val_choices, axis=0)

    return [training, validation, testing]

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


def unet_main_block(m, n_filters, dim, bn, do_rate):
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
