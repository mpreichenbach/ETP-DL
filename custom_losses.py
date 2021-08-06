import tensorflow as tf

def iou(y_true, y_pred):
    # Given input tensors of shape (batch_size, height, width, n_classes), this computes the IoU-score for each class,
    # then averages over the classes to get an overall score.

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.math.multiply(y_true, y_pred)
    i_totals = tf.reduce_sum(intersection, axis=(1, 2))
    union = tf.math.add(y_true, y_pred)
    u_totals = tf.reduce_sum(union, axis=(1, 2))

    # Note that some references insist on adding 1 to the numerator and denominator, or an epsilon to the denominator,
    # to avoid division by zero. This is important when the IoU is used to gauge accuracy of bounding box predictions in
    # object-detection tasks, because it's possible that the true and predicted bounding boxes are both empty (i.e.,
    # when the network predicts a true negative). But in semantic segmentation tasks this is unnecessary, since the
    # every voxel in the ground-truth and predicted tensors has at least one nonzero entry.
    cat_iou = i_totals / u_totals
    avg_iou = tf.reduce_mean(cat_iou)

    return 1 - avg_iou

def dice(y_true, y_pred):
    a = 1 + 1

    return a
