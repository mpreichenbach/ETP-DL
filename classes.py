from datasets import data_generator
from helper_functions import pt_model, vec_to_label
import numpy as np
import os
import pandas as pd
from sklearn.metrics import jaccard_score, recall_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

class SemSeg():
    def __init__(self):
        # model
        self.model = None
        self.weight_path = None
        self.n_classes = 0
        self.class_names = []

        # training data
        self.training_data = None
        self.training_path = None
        self.n_training_examples = 0
        self.batch_size = 0
        self.image_dimension = 512

        # validation data
        self.validation_data = None
        self.validation_path = None
        self.n_validation_examples = 0

    def initial_model(self, backbone="VGG19", n_filters=16, n_classes=2, class_names=("non-building", "building"), concatenate=True,
                   dropout=0.2, optimizer='Adam', loss='categorical_crossentropy'):

        """Initializes a model with pretrained weights in the downsampling path from 'backbone'.
            Args:
                backbone (str): the name of the model used for backbones; see pt_model() for valid strings;
                n_classes (int > 1): the number of classes to predict"""

        self.model = pt_model(backbone=backbone,
                              n_filters=n_filters,
                              n_classes=n_classes,
                              concatenate=concatenate,
                              do=dropout,
                              opt=optimizer,
                              loss=loss)

        self.n_classes = n_classes
        self.class_names = class_names

        print("Model weights initialized.")

    def load_weights(self, path):
        """After initializing a model, this loads pretrained weights to the upsampling path.
            Args:
                path (str): the path to the directory containing pretrained weights."""

        if (self.model == None):
            raise Exception("Initialize model with initialize_weights() method before loading weights.")

        self.model.load_weights(path)
        self.weight_path = path

    def load_generator(self, train_path, val_path=None, image_dim=512, batch_size=8, one_hot=True):
        """Creates generators for training and validation data. Calls the function data_generator, which has arguments
        arguments for data augmentation (flipping and rotating) not included here.

        Args:
            train_path (str): the path to the training data, should have 'rgb' and 'masks' subfolders;
            val_path (str): the path to the validation data, should have 'rgb' and 'masks' subfolders;
            image_dim (int): the height of images in the training data (assumed to be square);
            batch_size (int > 0): the size of the batches to generate;
            one_hot (bool): whether to perform a one-hot encoding on the labeled imagery."""

        # make sure that the number of rgb and mask images is the same before training updating attributes
        if len(os.listdir(train_path + "rgb")) == len(os.listdir(train_path + "masks")):
            self.training_data = data_generator(image_dir=os.path.join(train_path, "rgb"),
                                                mask_dir=os.path.join(train_path, "masks"),
                                                batch_size=batch_size,
                                                classes=self.n_classes,
                                                one_hot=one_hot)
            self.n_training_examples = len(os.listdir(train_path + "rgb"))
            self.training_path = train_path
            self.image_dimension = image_dim
            self.batch_size = batch_size

            if val_path is None:
                print("Training data generator loaded.")

        else:
            raise Exception("The number of RGB images does not match the number of masks in the training data.")

        if val_path is not None:
            # make sure that the number of RGB and mask images is the same before updating validation attributes
            if len(os.listdir(val_path + "rgb")) == len(os.listdir(val_path + "masks")):
                self.n_validation_examples = len(os.listdir(val_path + "rgb"))

            else:
                raise Exception("The number of RGB images does not match the number of Mask images in the validation "
                                "data.")

            self.validation_data = data_generator(image_dir=os.path.join(val_path, "rgb"),
                                                  mask_dir=os.path.join(val_path, "masks"),
                                                  batch_size=batch_size,
                                                  classes=self.n_classes,
                                                  one_hot=one_hot)
            self.validation_path = val_path

            print("Training and validation data generators loaded.")

    def train_model(self, epochs, save_path=None, monitor='val_loss', lr_factor=0.2, lr_patience=50,
                    my_callbacks=[], verbose=1):
        """Fits the model with options for weight-saving, learning rate reduction, csv-logging, and other callbacks.
            Args:
                epochs (int): the number of epochs to train the model for;
                save_path (str): the directory in which to save weights of the best model (based on val_loss);
                lr_monitor (str): one of 'val_loss' or 'loss';
                lr_factor (0 < float < 1): the factor by which to reduce the learning rate;
                lr_patience (int): the number of epochs to elapse before reducing the learning rate;
                verbose (0, 1, 2): the level of verbosity for fitting (see the Keras model.fit() method).)"""

        if monitor == 'val_loss' and self.validation_data is None:
            raise Exception("Either load validation_data, or change lr_monitor to 'loss'.")

        # instantiate the callbacks for the model.fit() method
        if save_path is not None:
            csv_callback = CSVLogger(save_path + "training_log.csv", append=True)
            my_callbacks.append(csv_callback)

            checkpoint_callback = ModelCheckpoint(filepath=save_path, monitor=monitor, save_best_only=True,
                                                  save_weights_only=True, save_freq="epoch", verbose=verbose)
            my_callbacks.append(checkpoint_callback)

        if monitor is not None and lr_factor is not None and lr_patience is not None:
            lr_callback = ReduceLROnPlateau(monitor=monitor, factor=lr_factor, patience=lr_patience)
            my_callbacks.append(lr_callback)

        # calculate the number of steps_per_epoch to exhaust the training/validation generators
        steps_per_epoch = int(self.n_training_examples / self.batch_size) + 1
        validation_steps = int(self.n_validation_examples / self.batch_size) + 1

        self.model.fit(self.training_data,
                       epochs=epochs,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=self.validation_data,
                       validation_steps=validation_steps,
                       callbacks=my_callbacks,
                       verbose=verbose)


class Metrics():
    def __init__(self):
        self.model = None
        self.test_data = []
        self.class_names = ["not-building", "building"]
        self.predicted_data = None
        self.metrics = None
        self.confusion_table = None

    def predict_test_data(self, verbose=50):
        """Uses the initialized model to do inference on the manually loaded test data.
        Args:
            verbose (int): updates on progress are printed on multiples of this number."""

        if len(self.test_data) !=2:
            raise Exception("Load test data as a list of the form [rgb, masks].")

        rgb = self.test_data[0]
        masks = self.test_data[1]

        holder = np.zeros(masks.shape, dtype=np.uint8)

        for i in range(len(masks)):
            holder[i] = vec_to_label(self.model.predict(np.expand_dims(rgb[i], 0)))
            if (i + 1) % verbose == 0:
                print("Finished inference on image " + str(i) + "/" + str(len(rgb)) + ".")

        self.predicted_data = holder
        print("Finished inference on test data.")

    def generate_metrics(self, metrics=True, confusion_table=True):
        """Generates performance metrics for the loaded model on the test data, which must be loaded manually into the
        test_data attribute.

        Args:
            metrics (bool): whether to calculate metrics like accuracy, IoU, and Dice scores;
            confusion_table (bool): whether to generate a confusion table."""

        if self.predicted_data is None:
            raise Exception("Run predict_test_data method before generating metrics.")

        if len(self.class_names) != len(np.unique(self.test_data[1])):
            raise Exception("Mismatch between number of class names, and number of classes present in the test data.")
        else:
            precision_names = [name + " precision" for name in self.class_names]
            recall_names = [name + " recall" for name in self.class_names]
            iou_names = [name + "IoU Score" for name in self.class_names]
            dice_names = [name + "IoU Score" for name in self.class_names]

        y_true = self.test_data[1].flatten()
        y_pred = self.predicted_data.flatten()

        if metrics:
            df_precision = pd.DataFrame(columns=precision_names)
            df_recall = pd.DataFrame(columns=recall_names)
            df_iou = pd.DataFrame(columns=iou_names)
            df_dice = pd.DataFrame(columns=dice_names)

            # compute precision and recall scores
            for i in range(len(self.class_names)):
                df_precision[1, i] = precision_score(y_true, y_pred, pos_label=i)
                df_recall[1, i] = recall_score(y_true, y_pred, pos_label=i)
                df_iou[1, i] = jaccard_score(y_true, y_pred, pos_label=i)
                df_dice[1, i] = f1_score(y_true, y_pred, pos_label=i)

            self.metrics = pd.concat([df_precision, df_recall, df_iou, df_dice], axis=1)

            print("Metrics generated.")

        if confusion_table:
            table = confusion_matrix(y_true.flatten(), y_pred.flatten(), normalize='true').round(2)
            self.confusion_table = pd.DataFrame(table, index=self.class_names, columns=self.class_names)

            print("Confusion table generated.")

    def view_tiles(self, n=5, idx=None, rgb=True, mask=True, pred=True, data=None):
        """Outputs an image for visually inspecting model performance. Defaults to a random selection from the test
        data, unless a tuple (RGB, masks) of Numpy arrays is passed in data.

        Args:
            n (int): number of examples to show (number of rows in the image);
            idx (list): a vector of indices in the test set;
            rgb (boolean): whether to include RGB imagery in the display;
            mask (boolean): whether to include masks in the display;
            pred (boolean): whether to include predictions in the display;
            data (tuple): a tuple (RGB, masks) of numpy arrays to draw from."""

        if idx is None and data is not None:
            raise Exception("Specify indices to display from the data.")

        if idx is not None and data is None:
            raise Exception("Include a tuple (RGB, masks) of data to draw from.")

        ncol = np.sum(np.where([rgb, mask, pred], 1, 0))

