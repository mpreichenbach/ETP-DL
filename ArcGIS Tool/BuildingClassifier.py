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
                'description': "INput model description (EMD) JSON file."
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
        # obtain the input image
        input_image = pixelBlocks['raster_pixels']

        # Todo: transform the input image to the shape of the model's input
        # input_image = np.transform(input_image, ....)

        # Todo: run the model on the transformed input image, something like
        # model_output = self.model.run(input_image)

        # Todo: wrap the classified raster in dictonary, something like
        # pixelBlocks['output_pixels'] =  model_output.astype(props['pixelType'], copy=False)

        # Return the dict
        return pixelBlocks
