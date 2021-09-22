import numpy as np

def iou(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the IoU-loss for each class,
    # and returns a vector of IoU scores. See https://en.wikipedia.org/wiki/Jaccard_index for more information.

    intersection = np.multiply(y_true, y_pred)
    i_totals = np.sum(intersection, axis=(0, 1, 2), dtype=np.uint8)
    union = y_true + y_pred - intersection
    u_totals = np.sum(union, axis=(0, 1, 2), dtype=np.uint8)

    # the following division will result in nans, so we suppress the warnings since we want to know where nans occur
    with np.errstate(all='ignore'):
        iou_vec = i_totals / u_totals

    return iou_vec

def dice(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the dice-loss for each class,
    # and returns a vector of dice scores. See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient for
    # more information.

    numerator = 2 * np.sum(np.multiply(y_true, y_pred), axis=(0, 1, 2))
    denominator = np.sum(y_true, axis=(0, 1, 2)) + np.sum(y_pred, axis=(0, 1, 2))

    # the following division can result in nans, so we suppress the warnings since we want to know where nans occur
    with np.errstate(all='ignore'):
        dice_vec = numerator / denominator

    return dice_vec

def total_acc(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the percentage accuracy of the
    # model's prediction for each class.

    # these two lines set the prediction's max depth value as 1, and every other value as 0
    comparison = np.equal(np.amax(y_pred, axis=-1, keepdims=True), y_pred)
    y_pred_oh = np.where(comparison, 1, 0).astype(np.uint8)

    intersection = np.multiply(y_true, y_pred_oh)
    numerator = np.sum(intersection, axis=(0, 1, 2))
    denominator = np.sum(y_true, axis=(0, 1, 2))

    with np.errstate(all='ignore'):
        acc_vec = numerator / denominator

    return acc_vec
