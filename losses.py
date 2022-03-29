from tensorflow.keras import backend as K


def iou_loss(y_true, y_pred):
    """Computes the loss 1 - IoU; note that this function as written only works on binary data.
    Args:
        y_true: tensor of ground-truth values of size (batch, height, width), so not one-hot encoded;
        y_pred: tensor of model predictions of size (batch, height, width, 2), so one-hot encoded."""

    # ensures no division by zero, which can occur when a model accurately predicts no instances of the class.
    smooth = 1

    intersection = K.cumsum(K.flatten(y_true * y_pred[-1]))
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred[-1])) + smooth

    iou = intersection / union

    return 1 - iou

