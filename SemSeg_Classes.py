import os
import numpy as np
import pandas as pd
from helper_functions import pt_model, view_tiles
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from helper_functions import unet_main_block
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D

# vgg16_callbacks = [
#     TerminateOnNaN(),
#     ModelCheckpoint(filepath='Saved Models/2021-7-30/VGG16/', saved_best_only=True, save_weights_only=True),
#     EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, restore_best_weights=True),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_delta=0.0001),
#     CSVLogger('Saved Models/2021-7-30/CSV logs/VGG16.csv', append=True)
# ]
# vgg19_callbacks = [
#     TerminateOnNaN(),
#     ModelCheckpoint(filepath='Saved Models/2021-7-30/VGG19/', saved_best_only=True, save_weights_only=True),
#     EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, restore_best_weights=True),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_delta=0.0001),
#     CSVLogger('Saved Models/2021-7-30/CSV logs/VGG19.csv', append=True)
# ]

class DigitalGlobeDataset:
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
        self.pretrained_models = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2',
                                  'ResNet101V2', 'ResNet152V2', 'InceptionV3', 'InceptionResNetV2', 'MobileNet',
                                  'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile',
                                  'NASNetLarge', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                                  'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

    # if you want separate objects for the data, call this function
    def load_data(self, load_masks=True, ttv_split=True):
        if self.ir:
            sats = np.load(self.data_path + 'RGBIR_tiles_' + str(self.dim) + '.npy')
        else:
            sats = np.load(self.data_path + 'RGB_tiles_' + str(self.dim) + '.npy')

        if len(self.classes) == 1:
            enc = np.load(self.data_path + 'Binary Classification/' + str(self.classes[0]) + '_' + str(self.dim) + '.npy')
        else:
            enc = np.load(self.data_path + 'Encoded_tiles_' + str(self.dim) + '.npy')

        if load_masks:
            masks = np.load(self.data_path + 'Label_tiles_' + str(self.dim) + '.npy')

        if ttv_split:
            train_choices = np.load(self.data_path + 'Training_Choices_' + str(self.dim) + '.npy')
            val_choices = np.load(self.data_path + 'Validation_Choices_' + str(self.dim) + '.npy')

            sats_train = sats[train_choices]
            holder = np.delete(sats, train_choices, axis=0)
            sats_val = holder[val_choices]
            sats_test = np.delete(holder, val_choices, axis=0)

            if load_masks:
                masks_train = masks[train_choices]
                holder = np.delete(masks, train_choices, axis=0)
                masks_val = holder[val_choices]
                masks_test = np.delete(holder, val_choices, axis=0)

            enc_train = enc[train_choices]
            holder = np.delete(enc, train_choices, axis=0)
            enc_val = holder[val_choices]
            enc_test = np.delete(holder, val_choices, axis=0)

            if load_masks:
                return [sats_train, sats_val, sats_test, masks_train, masks_val, masks_test, enc_train, enc_val,
                        enc_test]
            else:
                return [sats_train, sats_val, sats_test, enc_train, enc_val]
        else:
            if load_masks:
                return [sats, masks, enc]
            else:
                return [sats, enc]



#####
# Untrained models
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