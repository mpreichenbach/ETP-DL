import json
import numpy as np
import tensorflow as tf

class BuildingClassifier:

    def __init__(self):
        self.name = "Building Classifier"
        self.description = ("This function uses a trained deep-learning model to detect buildings in an RGB image.")

    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        json_file = kwargs['model']
        with open(json_file, 'r') as f:
            self.json_info = json.load(f)

        model_path = self.json_info['ModelFile']

        self.model = tf.saved_model.load(model_path)

    def get_parameter_info(self):
        return [
            {
                'name': 'rgb_raster',
                'dataType': 'raster',
                'required': True,
                'value': None,
                'displayName': 'RGB Raster',
                'description': "The 3-band raster in which to detect buildings."
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': "Input Model Description (EMD) File",
                'description': "Input model description (EMD) JSON file."
            }
        ]

    # padding, tx, and ty may not be necessary
    def getConfiguration(self, **scalars):
        if 'padding' in scalars:
            self.padding = int(scalars['padding'])

        return {
            'extractBands': tuple(self.json_info['ExtractBands']),
            'padding': self.padding,
            'tx': self.json_info['ImageWidth'] - 2 * self.padding,
            'ty': self.json_info['ImageHeight'] - 2 * self.padding
        }

    # I don't understand the purpose of this method
    def getFields(self):
        fields = {
            'fields': [
                {
                    'name': 'OID',
                    'type': 'esriFieldTypeOID',
                    'alias': 'OID'
                },
                {
                    'name': 'Class',
                    'type': 'esriFieldTypeString',
                    'alias': 'Class'
                },
                {
                    'name': 'Confidence',
                    'type': 'esriFieldTypeDouble',
                    'alias': 'Confidence'
                },
                {
                    'name': 'Shape',
                    'type': 'esriFieldTypeGeometry',
                    'alias': 'Shape'
                }
            ]
        }
        return json.dumps(fields)

    # this is the workhorse method for the Classify Pixels Using Deep Learning tool
    def updatePixels(self, tlc, shape, props, **pixelBlocks):

        # this function converts one-hot encoded pixels to integer labels
        def vec_to_label(oh_array):
            output = np.argmax(oh_array, axis=-1).astype(np.uint8)

            return output

        # obtain the input image, initialize prediction holder
        input_image = pixelBlocks['raster_pixels']

        # dimensions of inference window (square) for the neural network
        d = 1024

        # compute appropriate padding amounts for the prediction holder
        output_padding = [0, 0]
        input_dims = input_image.shape[1:3]
        for i in range(2):
            if input_dims[i] % d == 0:
                continue
            else:
                b = input_dims[i]
                output_padding[i] = int(0.5 * ((d * (int(b / d) + 1)) - b))

        # initialize the prediction holder and pad the input
        ph, pw = output_padding
        padded_input = np.pad(input_image, ((0,), (ph,), (pw,)), mode='reflect')
        padded_predictions = np.zeros(padded_input.shape, dtype=np.uint8)

        # run inference on tiles of padded_input, place into prediction_holder
        n_tiles_height = int(padded_input / d)
        n_tiles_width = int(padded_input / d)
        padded_input = np.moveaxis(padded_input, 0, 2)

        for i in range(n_tiles_height):
            for j in range(n_tiles_width):
                input_tile = np.expand_dims(padded_input[(d * i):(d * (i + 1)), (d + j):(d * (j + 1)), ], 0)
                predicted_tile = np.squeeze(self.model.predict(input_tile))
                predicted_tile = vec_to_label(predicted_tile)

                padded_predictions[(d * i):(d * (i + 1)), (d + j):(d * (j + 1))] = predicted_tile

        # keep only the dimensions of the original input
        unpadded_predictions = padded_predictions[ph:-ph, pw:-pw]

        # put the output into the dictionary, and return it
        pixelBlocks['output_pixels'] = unpadded_predictions

        return pixelBlocks
