from datasets import data_generator
from helper_functions import pt_model
import os
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
        self.n_classes = 0

    def initialize_weights(self, backbone="VGG19", n_classes=2, concatenate=True, dropout=0.2, optimizer='Adam',
                           loss='categorical_crossentropy'):
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

        self.n_classes = n_classes

        print("Model weights initialized.")

    def load_weights(self, path):
        """After initializing a model, this loads pretrained weights to the upsampling path.
            Args:
                path (str): the path to the directory containing pretrained weights."""

        if (self.model == None):
            raise Exception("Initialize model with initialize_weights() method before loading weights.")

        self.model.load_weights(path)
        self.weight_path = path

    def load_generator(self, train_path, val_path=None, image_dim=512, batch_size=8, one_hot=True, ):
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

            print("Training data generator loaded.")

        else:
            raise Exception("The number of RGB images does not match the number of masks in the training data.")

        if val_path is not None:
            # make sure that the number of RGB and mask images is the same before updating validation attributes
            if len(os.listdir(val_path + "rgb")) == len(os.listdir(val_path + "masks")):
                self.n_validation_examples = len(os.listdir(val_path + "rgb/images/"))

            else:
                raise Exception("The number of RGB images does not match the number of Mask images in the validation "
                                "data.")

            self.validation_data = data_generator(image_dir=os.path.join(val_path, "rgb"),
                                                  mask_dir=os.path.join(val_path, "images"),
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
                                                  save_weights_only=True)
            my_callbacks.append(checkpoint_callback)

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