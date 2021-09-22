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
    # more information. Note that I used Tensorflow math functions to that this may be incorporated into a Keras losses

    numerator = 2 * np.sum(np.multiply(y_true, y_pred), axis=(0, 1, 2))
    denominator = np.sum(y_true, axis=(0, 1, 2)) + np.sum(y_pred, axis=(0, 1, 2))

    # the following division can result in nans, so we suppress the warnings since we want to know where nans occur
    with np.errstate(all='ignore'):
        dice_vec = numerator / denominator

    return dice_vec

def total_acc(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the total accuracy of the
    # model's prediction.

    y_pred = tf.where(
        tf.equal(tf.reduce_max(y_pred, axis=-1, keepdims=True), y_pred),
        tf.constant(1, shape=y_pred.shape), tf.constant(0, shape=y_pred.shape))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    intersection = tf.math.multiply(y_true, y_pred)
    numerator = tf.math.reduce_sum(intersection).numpy()
    denominator = tf.math.cumprod(y_true.shape).numpy()[-2]

    proportion = (numerator / denominator)

    return proportion
