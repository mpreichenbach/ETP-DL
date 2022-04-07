from tensorflow.keras import backend as K


def iou_loss(y_true, y_pred):
    """Computes a variation of the loss 1 - IoU, the difference being the inclusion of the 'smooth' parameter to ensure
     no division by zero, and a scaling parameter to ensure IoU scores in the range [0, 1].

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width, 2);
        y_pred: tensor of model predictions of size (batch, height, width, 2), so one-hot encoded."""

    # ensures no division by zero, which can occur when a model accurately predicts no instances of the class.
    smooth = 1

    intersection = 2 * K.sum(K.flatten(y_true * y_pred))
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + smooth

    # perfect overlap implies iou = 1 due to the smooth parameter value, perfect non-overlap implies iou = 0.
    iou = intersection / union

    return 1 - iou

def log_iou_loss(y_true, y_pred):
    """Compute a variation of the -log(IoU) loss introduced in 'Unitbox': An Advanced Object Detection Network. This
    version includes the 'smooth' parameter to ensure no division by zero.

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width), so not one-hot encoded;
        y_pred: tensor of model predictions of size (batch, height, width, 2), so one-hot encoded."""

    # ensures a nonzero numerator and denominator, so that the logarithm is well-define.
    smooth = 1

    intersection = 2 * K.sum(K.flatten(y_true * y_pred))
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + smooth

    return - K.log(intersection / union)
