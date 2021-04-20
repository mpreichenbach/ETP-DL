import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dropout, Input, Lambda, \
    MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model

# ----------------------------------------------------
# UNet
# ----------------------------------------------------

def main_block(m, filters, bn, do_rate):
    """The primary convolutional block in the UNet network"""

    n = Dropout(do_rate)(m)
    n = Conv2D(filters, 3, activation='relu', padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do_rate)(n) if do_rate else n
    n = Conv2D(filters, (3, 3), activation='relu', padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return n




def unet(im_dim, filters, bn=True, do_rate=0., opt='Adam', loss='mse', depth=4):
    """Implements the UNet architecture, with options for Dropout and BatchNormalization"""

    # Downsampling path
    input_img = Input(shape=(im_dim, im_dim, 1))
    b1 = main_block(input_img, filters=filters, bn=bn, do_rate=do_rate)
    max_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(b1)
    filters *= 2
    b2 = main_block(max_1, filters=filters, bn=bn, do_rate=do_rate)
    max_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(b2)
    filters *= 2
    b3 = main_block(max_2, filters=filters, bn=bn, do_rate=do_rate)
    max_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(b3)
    filters *= 2
    b4 = main_block(max_3, filters=filters, bn=bn, do_rate=do_rate)

    # Upsampling path
    up_1 = UpSampling2D(size=(2, 2))(b4)
    con_1 = Concatenate(axis=-1)([up_1, b3])
    filters = int(filters / 2)
    b5 = main_block(con_1, filters=filters, bn=bn, do_rate=do_rate)
    up_2 = UpSampling2D(size=(2, 2))(b5)
    con_2 = Concatenate(axis=-1)([up_2, b2])
    filters = int(filters / 2)
    b6 = main_block(con_2, filters=filters, bn=bn, do_rate=do_rate)
    up_3 = UpSampling2D(size=(2, 2))(b6)
    con_3 = Concatenate(axis=-1)([up_3, b1])
    filters = int(filters / 2)
    b7 = main_block(con_3, filters=filters, bn=bn, do_rate=do_rate)
    output_img = Conv2D(1, (1, 1), padding='same', activation='relu')(b7)

    # Creates a model and compiles with an optimizer and loss function ('Adam' and 'mse' are the defaults)
    cnn = Model(input_img, output_img)
    cnn.compile(optimizer=opt, loss=loss)

    return cnn


# ----------------------------------------------------
# AlexNet
# ----------------------------------------------------

