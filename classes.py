import numpy as np
from helper_functions import pt_model

class SemSeg():
    def __init__(self):
        self.model = None
        self.weight_path = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None

    def initialize_weights(self, backbone="VGG19", n_classes=2, concatenate=True, dropout=0.2, optimizer='Adam',
                           loss='sparse_categorical_crossentropy'):
        """Initializes a model with pretrained weights in the downsampling path from 'backbone'.
            Args:
                backbone (string): the name of the model used for backbones; see pt_model() for valid strings;
                n_classes (int > 1): the number of classes to predict"""

        self.model = pt_model(backbone=backbone, n_classes=n_classes, concatenate=concatenate, opt=optimizer, loss=loss)
        print("Model weights initialized.")

    def load_weights(self, path):
        """After initializing a model, this loads pretrained weights to the upsampling path.
            Args:
                path (string): the path to the directory containing pretrained weights."""

        if (self.model == None):
            raise Exception("Initialize model with initialize_weights() method before loading weights.")

        self.model.load_weights(path)
        self.weight_path = path

    def train(self):
