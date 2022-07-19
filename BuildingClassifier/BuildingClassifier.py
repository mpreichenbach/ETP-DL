from os import environ
# this line overrides the pixel limit that opencv (aka cv2) imposes on loaded images, and needs to be before tensorflow
environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
from tensorflow.keras.models import load_model
from cv2 import imread, IMREAD_COLOR, imwrite
from numpy import argmax, expand_dims, pad, squeeze, zeros


class BuildingClassifier:
    def __init__(self):
        self.rgb_raster = None
        self.predicted_raster = None
        self.model = None
        self.raster_size = ()
        self.pixel_type = None

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def load_rgb(self, rgb_path):
        self.rgb_raster = imread(rgb_path, IMREAD_COLOR)
        self.raster_size = self.rgb_raster.shape[0:2]
        self.pixel_type = self.rgb_raster.dtype

    def predict(self, window_size=1024):
        """Neural networks need a fixed window size during inference. The dimensions of the  input raster need to be
        padded so that each dimension is divisible by that window size; padding with 0's will cause edge-artifacts,
        so this method pads with values reflected from the image itself.

        Future note: Instead of having a user-specified (or even default) window-size argument, it would be best to
        dynamically calculate the largest window_size possible. This would take more development, because the largest
        window_size value is a function of the loaded model's memory footprint, and the tool-user's particular hardware.

         Args:
             window_size (int): the square window dimension for the model to do inference on."""

        # obtain the input image, initialize prediction holder
        input_image = self.rgb_raster

        # compute the appropriate padding amounts for the inference step.
        output_padding = [(0, 0), (0, 0)]
        input_dims = self.raster_size

        for i in range(len(input_dims)):
            if input_dims[i] % window_size == 0:
                continue
            else:
                dim = input_dims[i]
                exact_pad = 0.5 * ((window_size * (int(dim / window_size) + 1)) - dim)
                # if the difference between the padded and initial dimensions is odd, then exact_pad will have a nonzero
                # decimal part. In that case, we have to pad by different values on each side.
                if int(exact_pad) < exact_pad:
                    output_padding[i] = (int(exact_pad), int(exact_pad) + 1)
                else:
                    output_padding[i] = (int(exact_pad), int(exact_pad))

        # define the number of pixels to pad the input by
        height_pads, width_pads = output_padding

        # pad with reflected values from the image
        padded_input = pad(input_image, (height_pads, width_pads, (0, 0)), mode='reflect')

        # initialize an array to hold the predicted values
        padded_predictions = zeros(padded_input.shape[0:2], dtype=self.pixel_type)

        # run inference on tiles of padded_input, place into prediction_holder
        n_tiles_height = int(padded_input.shape[0] / window_size)
        n_tiles_width = int(padded_input.shape[1] / window_size)

        # this loop does inference on only one tile at a time, and could be (easily?) parallelized
        for i in range(n_tiles_height):
            for j in range(n_tiles_width):
                # this line gives the tile the shape (1, window_size, window_size, 3), which is necessary for inference
                input_tile = expand_dims(padded_input[(window_size * i):(window_size * (i + 1)),
                                                      (window_size * j):(window_size * (j + 1))], 0)

                # this line does inference, and yields a tile of shape (window_size, window_size, 2)
                predicted_tile = squeeze(self.model.predict(input_tile))

                # converts the class-probability vectors to a label
                predicted_tile = argmax(predicted_tile, axis=-1).astype(self.pixel_type)

                # updates the array which holds the predicted tiles
                padded_predictions[(window_size * i):(window_size * (i + 1)),
                                   (window_size * j):(window_size * (j + 1))] = predicted_tile

        # keep only the dimensions of the original input
        self.predicted_raster = padded_predictions[height_pads[0]:-height_pads[1], width_pads[0]:-width_pads[1]]

    def save_prediction(self, output_path):
        imwrite(output_path, self.predicted_raster)


if __name__ == "__main__":
    rgb_path = "Test Imagery/bellingham_clipped.tif"
    model_path = "VGG19 Inria/"
    output_path = "Example Imagery/bellingham_pred.tif"

    classifier = BuildingClassifier()
    classifier.load_model(model_path=model_path)
    classifier.load_rgb(rgb_path=rgb_path)
    classifier.predict()
    classifier.save_prediction(output_path=output_path)
