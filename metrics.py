import tensorflow as tf


def iou(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the IoU-loss for each class,
    # and returns a vector of IoU scores. See https://en.wikipedia.org/wiki/Jaccard_index for more information.
    # Note that I used Tensorflow math functions so that this may be incorporated into a Keras losses

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    intersection = tf.math.multiply(y_true, y_pred)
    i_totals = tf.reduce_sum(intersection, axis=(1, 2))
    union = tf.math.add(y_true, tf.math.add(y_pred, -intersection))
    u_totals = tf.reduce_sum(union, axis=(1, 2))

    # Note that some references insist on adding 1 (or small epsilon) to the numerator and denominator, to avoid
    # division by zero. This is important when the IoU is used to gauge accuracy of bounding box predictions in
    # object-detection tasks, because it's possible that the true and predicted bounding boxes are both empty (i.e.,
    # when the network predicts a true negative). But in semantic segmentation tasks this is unnecessary, since every
    # voxel in the ground-truth and predicted tensors has at least one nonzero entry.

    iou_vec = i_totals / u_totals
    scores = tf.reduce_mean(iou_vec, axis=0)

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
