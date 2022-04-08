import tensorflow as tf
from tensorflow.keras import backend as K

def iou_loss(y_true, y_pred):
    """Computes a variation of the loss 1 - IoU, the difference being the inclusion of the 'smooth' parameter to ensure
     no division by zero, and a scaling parameter to ensure IoU scores in the range [0, 1].

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width, 2);
        y_pred: tensor of model predictions of size (batch, height, width, 2), so one-hot encoded."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # ensures no division by zero, which can occur when a model accurately predicts no instances of the class.
    smooth = 1

    intersection = 2 * K.sum(K.flatten(y_true * y_pred))
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + smooth

    # perfect overlap yields a value slightly greater than 0, and perfect non-overlap yields a value slightly less than
    # 1, due to the inclusion of the smooth parameter
    final = 1 - intersection / union

    return final

def log_iou_loss(y_true, y_pred):
    """Compute a variation of the -log(IoU) loss introduced in 'Unitbox': An Advanced Object Detection Network. This
    version includes the 'smooth' parameter to ensure no division by zero.

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width), so not one-hot encoded;
        y_pred: tensor of model predictions of size (batch, height, width, 2), so one-hot encoded."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # ensures a nonzero numerator and denominator, so that the logarithm is well-defined.
    smooth = 1

    intersection = K.sum(K.flatten(y_true * y_pred)) + smooth
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + smooth

    # perfect overlap yields a value of 0, and perfect non-overlap yields a value greater than 1, which depends on the
    # dimensions of the image.
    final = - K.log(intersection / union)

    return final
