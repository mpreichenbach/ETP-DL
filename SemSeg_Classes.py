# contains the main classes used for model training and comparison
from helper_functions import unet_main_block, pt_model, vec_to_oh, oh_to_label, view_tiles
from matplotlib import pyplot as plt
from metrics import iou, dice, total_acc
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D
import time


class Metrics:
    """For the loaded model and test data attributes, has methods to compute metrics in nicely presentable formats."""

    # attributes
    def __init__(self, source, dimensions, res=None):
        """Args:
            source (str): one of 'Potsdam', 'Treadstone',
            dimensions (int): one of 256 or 512,
            res (int): one of 10, 50, 100, 200, giving the spacial resolution in cm."""

        self.confusion_tables = []
        self.data = []
        self.dimensions = dimensions
        self.lc_classes = ['Impervious Surface', 'Building', 'Low vegetation', 'High vegetation', 'Car', 'Clutter']
        self.models = []
        self.resolution = res
        self.score_table = pd.DataFrame(0, index=[], columns=[])
        self.source = source

        if source in ['Potsdam', 'Treadstone', 'Potsdam+Treadstone']:
            self.n_classes = 6

        self.class_prop = np.zeros(self.n_classes)

    # methods
    def load_models(self, backbones=None, losses='cc'):
        """Loads trained models with a given input dimensions.

        Args:
            backbones (list): a list of strings, with names taken from the list of pretrained backbones (see the
                            pretrained_weights attribute of SemSeg); defaults to ['VGG16', 'VGG19'],
            losses (list): list of strings giving the loss names of the corresponding models in the names list (can be
                            cc for categorical crossentropy, iou for Intersection-over-Union, or dice."""

        model_list = []
        loss_list = []

        if backbones is None:
            model_list = ['VGG16', 'VGG19']
        elif isinstance(backbones, list):
            model_list = backbones
        else:
            print('Please pass model name(s) as string (or list of strings).')

        if isinstance(losses, str):
            loss_list = []
            for name in model_list:
                loss_list.append(losses)
        elif isinstance(losses, list):
            loss_list = losses
        else:
            print('Please pass dimension(s) as string (or list of strings).')

        path = 'Saved Models/' + self.source + '/'
        n = len(model_list)

        for i in range(n):
            folder = model_list[i] + '_' + str(self.dimensions) + '_' + loss_list[i] + '/'
            dim_tuple = (self.dimensions, self.dimensions, 3)
            model = pt_model(model_list[i], dim_tuple, self.n_classes)

            model.load_weights(path + folder)
            model._name = folder[0:-1]
            self.models.append(model)

    def load_data(self):
        """Loads a test dataset on which to compute metrics. Note that one can also just set the test_data attribute
        directly."""
        if self.source == 'Potsdam' and self.resolution is None:
            folder = 'Data/Potsdam/Numpy Arrays/Test/'
            rgb = np.load(folder + 'Test_RGB_' + str(self.dimensions) + '.npy')
            labels = np.load(folder + 'Test_Labels_' + str(self.dimensions) + '.npy')
            enc = np.load(folder + 'Test_Encoded_' + str(self.dimensions) + '.npy')
        elif self.source == 'Potsdam' and self.resolution is not None:
            folder = 'Data/Potsdam/Numpy Arrays/Test/'
            rgb = np.load(folder + 'Test_RGB_' + str(self.dimensions) + '_' + str(self.resolution) + 'cm.npy')
            labels = np.load(folder + 'Test_Labels_' + str(self.dimensions) + '_' + str(self.resolution) + 'cm.npy')
            enc = np.load(folder + 'Test_Encoded_' + str(self.dimensions) + '_' + str(self.resolution) + 'cm.npy')
        elif self.source == 'Treadstone':
            folder = 'Data/Treadstone/'
            rgb = np.load(folder + 'RGB_' + str(self.dimensions) + '.npy')
            labels = np.load(folder + 'Labels_' + str(self.dimensions) + '.npy')
            enc = np.load(folder + 'Encoded_' + str(self.dimensions) + '.npy')

        self.data = [rgb, labels, enc]

    def make_scores(self, samples=None):
        """Sets self.score_table as a dataframe with predicted metrics for each model in self.models.

        Args:
            samples (int): the number of randomly chosen tiles from which to generate the metrics."""

        if samples is None and self.dimensions == 256:
            sample_size = 500
        elif samples is None and self.dimensions == 512:
            sample_size = 125
        else:
            sample_size = samples

        names = [x.name for x in self.models]
        iou_headers = ['IoU: ' + class_name for class_name in self.lc_classes]
        dice_headers = ['Dice: ' + class_name for class_name in self.lc_classes]
        acc_headers = ['Acc: ' + class_name for class_name in  self.lc_classes]

        choices = np.random.choice(len(self.data[0]), size=sample_size, replace=False)
        y_true = self.data[2][choices]
        rgb = self.data[0][choices]
        score_table = pd.DataFrame(0, index=names, columns=iou_headers + dice_headers)

        # in order to get accurate inferences times, we first need these lines to load the models into memory
        self.models[0].predict(rgb)
        self.models[1].predict(rgb)

        for i in range(len(self.models)):
            model = self.models[i]

            tic = time.perf_counter()
            y_pred = vec_to_oh(model.predict(rgb))
            toc = time.perf_counter()
            print('Finished predictions for model ' + model.name + ' ({}/{}).'.format(i + 1, len(self.models)))
            elapsed = toc - tic
            batch_time = round(100 * elapsed / len(y_pred), 2)

            n_pixels = sample_size * self.dimensions ** 2
            for j in range(self.n_classes):
                self.class_prop[j] = np.sum(y_pred[:, :, :, j]) / n_pixels

            iou_scores = iou(y_true, y_pred)
            weighted_iou = np.sum(self.class_prop * iou_scores)
            dice_scores = dice(y_true, y_pred)
            weighted_dice = np.sum(self.class_prop * dice_scores)
            acc_scores = total_acc(y_true, y_pred)
            weighted_acc = np.sum(self.class_prop * acc_scores)

            for j in range(len(iou_scores)):
                score_table.loc[model.name, acc_headers[j]] = acc_scores[j]
                score_table.loc[model.name, iou_headers[j]] = iou_scores[j]
                score_table.loc[model.name, dice_headers[j]] = dice_scores[j]

            score_table.loc[model.name, 'Weighted Accuracy'] = weighted_acc
            score_table.loc[model.name, 'Weighted IoU'] = weighted_iou
            score_table.loc[model.name, 'Weighted Dice'] = weighted_dice
            score_table.loc[model.name, 'GPU Inference Time'] = round(batch_time, 2)

        self.score_table = score_table

    def make_confusion(self, samples=None):
        """Sets self.confusion_tables as a list of confusion tables for each entry of self.models. Note that this
        process may take a long time if you have large datasets or many models to compare.

        Args:
            samples (int): the number of randomly chosen tiles from which to generate the confusion tables."""

        if samples is None and self.dimensions == 256:
            sample_size = 500
        elif samples is None and self.dimensions == 512:
            sample_size = 125
        else:
            sample_size = samples

        choices = np.random.choice(len(self.data[0]), size=sample_size, replace=False)
        y_true = self.data[1][choices]

        for i in range(len(self.models)):
            model = self.models[i]

            print('Converting predictions to integer labels on {} images.'.format(sample_size))
            y_pred = oh_to_label(vec_to_oh(model.predict(self.data[0][choices])))
            table = (100 * confusion_matrix(y_true.flatten(), y_pred.flatten(), normalize='true')).round(2)
            cm = pd.DataFrame(table, index=self.lc_classes, columns=self.lc_classes)
            print('Finished confusion matrix for model ' + model.name + ' ({}/{}).'.format(i + 1, len(self.models)))

            self.confusion_tables.append(cm)

    def view_predictions(self, n_tiles=4, choices=None, cmap='Accent'):
        """See notes for view_tiles() in helper_functions.py.

        Args:
            n_tiles (int): the number of tiles to view,
            choices (array-like): indices of tiles to pull from self.data; default selects randomly,
            cmap (string): the name of a Matplotlib colormap."""

        view_tiles(sats=self.data[0],
                   masks=self.data[1],
                   models=self.models,
                   n_tiles=n_tiles,
                   classes=self.n_classes,
                   choices=choices,
                   cmap=cmap)

    def make_and_save(self, path):
        """Runs make_scores(), make_confusion(), and view_predictions() with default arguments, and saves to the
        specified folder.

        Args:
            path (str): the path to the folder (which should not exist already)"""

        os.makedirs(path)

        # load models and data if not loaded yet
        if self.models == []:
            self.load_models()
        if self.data == []:
            self.load_data()

        self.make_scores()
        self.make_confusion()

        # save all the results
        self.score_table.to_csv(os.path.join(path, 'Score_table.csv'))

        for i in range(len(self.confusion_tables)):
            name = self.models[i].name + '.csv'
            self.confusion_tables[i].to_csv(os.path.join(path, name))

        self.view_predictions()
        plt.savefig(os.path.join(path, 'images.png'))
        plt.close()


class SemSeg:

    # attributes
    def __init__(self, source, dim,):
        """Loads a data and provides resources to load models to train for semantic segmentation.
        Args:
            source (str): member of the list ['Potsdam', 'Treadstone'],
            dim (int): member of the list [256, 512]."""

        self.dim = dim
        self.data_path = 'Data/' + source + '/Numpy Arrays/'
        self.class_df = pd.read_csv(self.data_path + 'class_df.csv')
        self.model_list = 'No models have been loaded.'

        # the following method is filled from https://keras.io/api/applications/ and may be updated in the future
        self.pretrained_models = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2',
                                  'ResNet101V2', 'ResNet152V2', 'InceptionV3', 'InceptionResNetV2', 'MobileNet',
                                  'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile',
                                  'NASNetLarge', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                                  'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']


    def load_data(self):
        rgb_train = np.load(self.data_path + 'Train/RGB_' + str(self.dim) + '.npy')
        rgb_val = np.load(self.data_path + 'Validation/RGB_' + str(self.dim) + '.npy')
        rgb_test = np.load(self.data_path + 'Test/RGB_' + str(self.dim) + '.npy')

        labels_train = np.load(self.data_path + 'Train/Labels_' + str(self.dim) + '.npy')
        labels_val = np.load(self.data_path + 'Validation/Labels_' + str(self.dim) + '.npy')
        labels_test = np.load(self.data_path + 'Test/Labels_' + str(self.dim) + '.npy')

        enc_train = np.load(self.data_path + 'Train/Encoded_' + str(self.dim) + '.npy')
        enc_val = np.load(self.data_path + 'Validation/Encoded_' + str(self.dim) + '.npy')
        enc_test = np.load(self.data_path + 'Test/Encoded_' + str(self.dim) + '.npy')

        return [rgb_train, rgb_val, rgb_test, labels_train, labels_val, labels_test, enc_train, enc_val, enc_test]


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
