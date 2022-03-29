import numpy as np
from helper_functions import pt_model
from datasets import dataset_gen

class SemSeg():
    def __init__(self):
        self.model = None
        self.weight_path = None
        self.training_data = None
        self.validation_data = None

    def initialize_weights(self, backbone="VGG19", n_classes=2, concatenate=True, dropout=0.2, optimizer='Adam',
                           loss='sparse_categorical_crossentropy'):
        """Initializes a model with pretrained weights in the downsampling path from 'backbone'.
            Args:
                backbone (str): the name of the model used for backbones; see pt_model() for valid strings;
                n_classes (int > 1): the number of classes to predict"""

        self.model = pt_model(backbone=backbone,
                              n_classes=n_classes,
                              concatenate=concatenate,
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

        self.training_data = dataset_gen(dim=image_dim,
                                         batch_size=batch_size,
                                         rgb_path = train_path + "rgb/",
                                         )

    def train(self):
