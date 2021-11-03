import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import xception, vgg16, vgg19, resnet, resnet_v2
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D


def pt_model(backbone, n_classes, concatenate=True, do=0.2, opt='Adam', loss='categorical_crossentropy'):
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
    input = Input((None, None, 3), dtype=tf.float32)

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
        # note that this path has no feature concatenation from the downsampling path; the Xception architecture doesn't
        # have obvious spots to do this, but I will continue thinking about it.
        filters = int(model_pt.output.shape[-1])
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        # x = Concatenate(axis=-1)([x, model_pt.layers[-5].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        # x = Concatenate(axis=-1)([x, model_pt.layers[-9].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        # x = Concatenate(axis=-1)([x, model_pt.layers[-13].output]) if concatenate else x
        filters = int(filters / 2)
        x = unet_main_block(x, n_filters=filters, dim=3, bn=True, do_rate=do)
        x = UpSampling2D(size=(2, 2))(x)
        # x = Concatenate(axis=-1)([x, model_pt.layers[-16].output]) if concatenate else x
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
        output_img = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

        # compile the model with the chosen optimizer and loss functions
        cnn_pt = Model(input, output_img)
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

def oh_to_label(oh_array):
    output = np.argmax(oh_array, axis=-1).astype(np.uint8)

    return output

def label_to_oh(label_array, n_classes):
    # is the following line wrong? test this
    holder = np.zeros(label_array.shape + (n_classes,), dtype=np.uint8)

    for i in range(n_classes):
        arr_slice = np.where(label_array == i, 1, 0)
        holder[:, :, :, i] = arr_slice

    holder = holder.astype(np.uint8)

    return holder

def tile_apply(image, model, overlap=0.0, mode='mean', conc=False):
    """Tiles an image, model predicts each tile, returns labeled image with original dimensions.

    Args:
        image (file object): an image-file that can be input into np.asarray,
        model (Keras model): the model with which to do the predicting,
        overlap (0.0 <= float < 1.0): the amount of overlap between tiles (note that 1.0 is impossible),
        mode (string): one of 'mean', 'max'; the method of resolving the labels in overlapping regions,
        conc (Boolean): whether to concatenate tiles before prediction (this is mostly for testing)."""

    arr = np.asarray(image)
    tile_dim = model.layers[0].input_shape[0][1]


    holder = np.zeros()

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
        arr = oh_to_label(vec_to_oh(model.predict(s_choices))).astype(np.uint8)
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
