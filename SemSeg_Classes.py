import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import confusion_matrix
from metrics import iou_loss, dice_loss, total_acc
from helper_functions import unet_main_block, pt_model, vec_to_oh, oh_to_label
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D


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


class Metrics:
    """For the loaded model and test data attributes, has methods to compute metrics in nicely presentable formats. Test
    data can be either one-hot encoded, or have integer labels."""

    # attributes
    def __init__(self, source, dimensions):
        """Args:
            project (str): one of 'Potsdam', 'Treadstone',
            dimensions (int): one of 256 or 512. """

        self.confusion_matrices = []
        self.data = []
        self.dimensions = dimensions
        self.models = []
        self.n_classes = 6
        self.score_table = pd.DataFrame(0, index=[], columns=[])
        self.source = source
        self.lc_classes = ['Impervious Surface', 'Building', 'Low vegetation', 'High vegetation', 'Car', 'Clutter']

    # methods
    def load_models(self, names, dimensions, losses='cc'):
        """Loads trained models with a given input dimensions.

        Args:
            names (list): a list of strings, with names taken from the list of pretrained backbones (see the
                            pretrained_weights attribute of SemSeg),
            dimensions (list): list of integers giving the dimensions of the input image of the corresponding model in
                            the names list,
            losses (list): list of strings giving the loss names of the corresponding models in the names list (can be
                            cc for categorical crossentropy, iou for Intersection-over-Union, or dice."""

        if type(names) == 'str':
            model_list = [names]
        elif type(names) == 'list':
            model_list = names
        else:
            print('Please pass model name(s) as string (or list of strings).')

        if type(dimensions) == 'int':
            dim_list = [dimensions]
        elif type(dimensions) == 'list':
            dim_list = dimensions
        else:
            print('Please pass dimension(s) as integer (or list of integers).')

        if type(losses) == 'str':
            loss_list = []
            for name in model_list:
                loss_list.append(losses)
        elif type(losses) == 'list':
            loss_list = losses
        else:
            print('Please pass dimension(S) as string (or list of strings).')

        path = 'Saved Models/Fully Trained Models/'
        n = len(model_list)

        for i in range(n):
            folder = model_list[i] + '_' + dim_list[i] + '_' + loss_list[i]
            dim_tuple = (dim_list[i], dim_list[i], 3)
            model = pt_model(model_list[i], dim_tuple, self.n_classes)

            model.load_weights(path + folder)
            model.name = folder
            self.models.append(model)

    def load_data(self, data):
        """Loads a test dataset on which to compute metrics. Note that one can also just set the test_data attribute
        directly."""
        if self.source == 'Potsdam':
            folder = 'Data/Potsdam/Numpy Arrays/Test/'
            rgb = np.load(folder + 'Test_RGB_' + str(self.dimensions) + '.npy')
            labels = np.load(folder + 'Test_Labels_' + str(self.dimensions) + '.npy')
            enc = np.load(folder + 'Test_Labels_' + str(self.dimensions) + '.npy')
        elif self.source == 'Treadstone':
            folder = 'Data/Treadstone/'
            rgb = np.load(folder + 'RGB_' + str(self.dimensions) + '.npy')
            labels = np.load(folder + 'Labels_' + str(self.dimensions) + '.npy')
            enc = np.load(folder + 'Encoded_' + str(self.dimensions) + '.npy')

        self.data = [rgb, labels, enc]

    def make_scores(self, sample_size=500):
        """Sets self.score_table as a dataframe with predicted metrics for each model in self.models.

        Args:
            sample_size (int): the number of randomly chosen tiles from which to generate the metrics."""

        names = [x.name for x in self.models]
        choices = np.random.choice(len(self.data[0]), size=sample_size, replace=False)
        y_true = self.data[2][choices]
        table = pd.DataFrame(0, index=names, columns=['Accuracy', 'IoU', 'Dice', 'GPU Inference Time'])

        for i in range(len(self.models)):
            model = self.models[i]

            tic = time.perf_counter()
            y_pred = model.predict(self.data[0][choices])
            toc = time.perf_counter()
            print('Finished predictions for model ' + model.name + '({}/{}).'.format(i + 1, len(self.models)))
            elapsed = toc - tic
            batch_time = round(100 * elapsed / len(y_pred), 2)

            table.loc[model.name, 'Accuracy'] = total_acc(y_true, y_pred)
            table.loc[model.name, 'IoU'] = - iou_loss(y_true, y_pred) + 1
            table.loc[model.name, 'Dice'] = - dice_loss(y_true, y_pred) + 1
            table.loc[model.name, 'GPU Inference Time'] = batch_time

        self.score_table = table

    def make_confusion(self, sample_size=500):
        """Sets self.confusion_tables as a list of confusion tables for each entry of self.models. Note that this
        process may take a long time if you have large datasets or many models to compare.

        Args:
            sample_size (int): the number of randomly chosen tiles from which to generate the confusion tables."""

        choices = np.random.choice(len(self.data[0]), size=sample_size, replace=False)
        y_true = self.data[1][choices]

        for i in range(len(self.models)):
            model = self.models[i]
            y_pred = oh_to_label(vec_to_oh(model.predict(self.data[0][choices])), dim=self.n_classes)
            table = confusion_matrix(y_true, y_pred, labels=self.lc_classes, normalize='true')
            print('Finished confusion matrix for model ' + model.name + '({}/{}).'.format(i + 1, len(self.models)))

            self.confusion_matrices.append(table)













class SemSeg:
    """Loads various data, compiles and fits NN models with options for pretraining, and functions to view results."""

    # attributes
    def __init__(self, dim, ir=False, binary_class=None):
        self.dim = dim
        self.ir = ir
        self.data_path = 'Data/Potsdam/Numpy Arrays/'
        self.class_df = pd.read_csv(self.data_path + 'class_df.csv')
        self.model_list = 'No models have been loaded.'

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
    def load_data(self, load_masks=True, ttv_split=True, test_only=False):
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

            if test_only:
                return [sats_test, masks_test, enc_test]

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

# ap_imgs = np.zeros((17, 3456, 4608), dtype=np.uint8)
# ap_onehot = np.zeros((17, 3456, 4608, 6), dtype=np.uint8)
# fc_imgs = np.zeros((16, 3000, 4000), dtype=np.uint8)
# fc_onehot = np.zeros((16, 3000, 4000, 6), dtype=np.uint8)
#
# for string in folders:
#     folder_path = 'Data/FC_AP_CC/' + string + '/labels/'
#     files = os.listdir(folder_path)
#     for i in range(len(files)):
#         im = Image.open(folder_path + files[i])
#         arr = np.asarray(im)
#         ap_imgs[i] = arr
#
#     if string == 'AP':
#         for n in range(17):
#             print('Working on image ' + str(n) + '/' + str(17) + '.')
#             for i in range(3456):
#                 for j in range(4608):
#                     ap_onehot[n, i, j] = cc_dict[ap_imgs[n, i, j]]
#
#     if string == 'FC':
#         for n in range(16):
#             print('Working on image ' + str(n) + '/' + str(16) + '.')
#             for i in range(3000):
#                 for j in range(4000):
#                     fc_onehot[n, i, j] = cc_dict[ap_imgs[n, i, j]]