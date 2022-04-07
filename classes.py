from datasets import dataset_gen
from helper_functions import pt_model
from losses import iou_loss, log_iou_loss
from os import listdir
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

class SemSeg():
    def __init__(self):
        # model attributes
        self.model = None
        self.weight_path = None

        # data attributes
        self.training_data = None
        self.training_path = None
        self.n_training_examples = 0
        self.validation_data = None
        self.validation_path = None
        self.n_validation_examples = 0
        self.data_batch_size = 0
        self.image_dimension = 512


    def initialize_weights(self, backbone="VGG19", n_classes=2, concatenate=True, dropout=0.2, optimizer='Adam',
                           loss='sparse_categorical_crossentropy'):
        """Initializes a model with pretrained weights in the downsampling path from 'backbone'.
            Args:
                backbone (str): the name of the model used for backbones; see pt_model() for valid strings;
                n_classes (int > 1): the number of classes to predict"""

        self.model = pt_model(backbone=backbone,
                              n_classes=n_classes,
                              concatenate=concatenate,
                              do=dropout,
                              opt=optimizer,
                              loss=loss)

        print("Model weights initialized.")

    def load_weights(self, path):
        """After initializing a model, this loads pretrained weights to the upsampling path.
            Args:
                path (str): the path to the directory containing pretrained weights."""

        if (self.model == None):
            raise Exception("Initialize model with initialize_weights() method before loading weights.")

        self.model.load_weights(path)
        self.weight_path = path

    def load_data(self, train_path, val_path=None, image_dim=512, batch_size=8):
        """Creates paired iterators for training and validation data. Calls the function dataset_gen(), which has
        more arguments than are given here.

        Args:
            train_path (str): the path to the training data, should have 'rgb' and 'masks' subfolders;
            val_path (str): the path to the validation data, should have 'rgb' and 'masks' subfolders;
            batch_size (int > 0): the size of the batches to generate."""

        self.training_path = train_path
        self.validation_path = val_path
        self.image_dimension = 512
        self.data_batch_size = 8

        self.training_data = dataset_gen(dim=image_dim,
                                         batch_size=batch_size,
                                         rgb_path=train_path + "rgb/",
                                         mask_path=train_path + "masks/")

        if val_path is None:
            print("Training data generator loaded.")

        # make sure that the number of rgb and mask images is the same before updating n_training_examples
        if len(listdir(train_path + "rgb/images/")) == len(listdir(train_path + "masks/images/")):
            self.n_training_examples = len(listdir(train_path + "rgb/images/"))
        else:
            raise Exception("The number of RGB images does not match the number of Mask images in the training data.")

        if val_path is not None:
            self.validation_data = dataset_gen(dim=image_dim,
                                               batch_size=batch_size,
                                               rgb_path=val_path + "rgb/",
                                               mask_path=val_path + "masks/")

            print("Training and validation data generators loaded.")

            # make sure that the number of rgb and mask images is the same before updating n_validation_examples
            if len(listdir(val_path + "rgb/images/")) == len(listdir(val_path + "masks/images/")):
                self.n_validation_examples = len(listdir(val_path + "rgb/images/"))
            else:
                raise Exception("The number of RGB images does not match the number of Mask images in the validation "
                                "data.")

    def train_model(self, epochs, save_path=None, monitor='val_loss', lr_factor=0.2, lr_patience=50,
                    my_callbacks=[], verbose=1):
        """Fits the model with options for weight-saving, learning rate reduction, csv-logging, and other callbacks.
            Args:
                epochs (int): the number of epochs to train the model for;
                save_path (str): the directory in which to save weights of the best model (based on val_loss);
                lr_monitor (str): one of 'val_loss' or 'loss';
                lr_factor (0 < float < 1): the factor by which to reduce the learning rate;
                lr_patience (int): the number of epochs to elapse before reducing the learning rate;
                verbose (0, 1, 2): the level of verbosity for the model.fit() method."""

        if monitor == 'val_loss' and self.validation_data is None:
            raise Exception("Either load validation_data, or change lr_monitor to 'loss'.")

        # instantiate the callbacks for the model.fit() method
        if save_path is not None:
            csv_callback = CSVLogger(save_path + "training_log.csv", append=True)
            my_callbacks.append(csv_callback)

            saving_callback = ModelCheckpoint(filepath=save_path, monitor=monitor, save_best_only=True,
                                            save_weights_only=True)
            my_callbacks.append(saving_callback)

        if monitor is not None and lr_factor is not None and lr_patience is not None:
            lr_callback = ReduceLROnPlateau(monitor=monitor, factor=lr_factor, patience=lr_patience)
            my_callbacks.append(lr_callback)

        # calculate the number of steps_per_epoch to exhaust the training/validation generators
        steps_per_epoch = int(self.n_training_examples / self.data_batch_size) + 1
        validation_steps = int(self.n_validation_examples / self.data_batch_size) + 1

        self.model.fit(self.training_data,
                       epochs=epochs,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=self.validation_data,
                       validation_steps=validation_steps,
                       callbacks=my_callbacks,
                       verbose=verbose)
