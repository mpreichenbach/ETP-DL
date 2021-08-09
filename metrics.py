import tensorflow as tf

def iou_loss(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the IoU-loss for each class,
    # then averages over the classes to get an overall score.
    # https://en.wikipedia.org/wiki/Jaccard_index

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    intersection = tf.math.multiply(y_true, y_pred)
    i_totals = tf.reduce_sum(intersection, axis=(1, 2))
    union = tf.math.add(y_true, tf.math.add(y_pred, -intersection))
    u_totals = tf.reduce_sum(union, axis=(1, 2))

    # Note that some references insist on adding 1 to the numerator and denominator, or an epsilon to the denominator,
    # to avoid division by zero. This is important when the IoU is used to gauge accuracy of bounding box predictions in
    # object-detection tasks, because it's possible that the true and predicted bounding boxes are both empty (i.e.,
    # when the network predicts a true negative). But in semantic segmentation tasks this is unnecessary, since the
    # every voxel in the ground-truth and predicted tensors has at least one nonzero entry.
    cat_iou = i_totals / u_totals
    avg_iou = tf.reduce_mean(cat_iou)

    return 1 - avg_iou

def dice_loss(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the dice-loss for each class,
    # then averages over the classes to get an overall score.
    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    numerator = 2 * tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=(1, 2))
    denominator = tf.reduce_sum(y_true, axis=(1, 2)) + tf.reduce_sum(y_pred, axis=(1, 2))

    # As in iou() above, we don't need to enforce nonzero denominators
    cat_dice = numerator / denominator
    avg_dice = tf.reduce_mean(cat_dice)

    return 1 - avg_dice

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

