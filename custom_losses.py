import tensorflow as tf

def iou(y_true, y_pred):
    # performance may be better without the following line
    # pred_max = tf.argmax(y_pred, axis=-1)
    true_flat = tf.reshape(y_true, [-1])
    pred_flat = tf.reshape(pred_max, [-1])

    intersection = tf.reduce_sum(tf.cast(true_flat, tf.float32) * tf.cast(pred_flat, tf.float32))
    union = tf.reduce_sum(tf.cast(true_flat, tf.float32) + tf.cast(pred_flat, tf.float32)) - intersection

    # Note that some references insist on adding 1 to the numerator and denominator, or an epsilon to the denominator,
    # to avoid division by zero. This is important when the IoU is used to gauge accuracy of bounding box predictions in
    # object-detection tasks, because it's possible that the true and predicted bounding boxes are both empty (i.e.,
    # when a particular object does not exist in an image, and the network predicts this fact correctly). But in
    # semantic segmentation tasks this is unnecessary, since the flattened y_true array will always have 1's.

    iou = intersection / union

    return 1 - iou

def dice(y_true, y_pred):
    a = 1 + 1

    return a
