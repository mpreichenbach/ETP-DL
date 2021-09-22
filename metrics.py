import numpy as np

def iou(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the IoU-loss for each class,
    # and returns a vector of IoU scores. See https://en.wikipedia.org/wiki/Jaccard_index for more information.

    # y_true = y_true.astype(np.float32)
    # y_pred = y_pred.astype(np.float32)

    intersection = np.multiply(y_true, y_pred)
    i_totals = np.sum(intersection, axis=(0, 1, 2), dtype=np.uint8)
    union = y_true + y_pred - intersection
    u_totals = np.sum(union, axis=(0, 1, 2), dtype=np.uint8)

    # the following division will result in nans, so we suppress the warnings since they're ignored later
    with np.errstate(divide='ignore'):
        iou_vec = i_totals / u_totals

    # to ignore than nans in the total, we use np.nansum()
    scores = np.nansum(iou_vec, axis=0)

    return scores

def dice(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the dice-loss for each class,
    # and returns a vector of dice scores. See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient for
    # more information. Note that I used Tensorflow math functions to that this may be incorporated into a Keras losses

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    numerator = 2 * tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=(1, 2))
    denominator = tf.reduce_sum(y_true, axis=(1, 2)) + tf.reduce_sum(y_pred, axis=(1, 2))

    # As in iou() above, we don't need to enforce nonzero denominators
    vec = numerator / denominator
    scores = tf.reduce_mean(vec, axis=0)

    return scores

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
