import numpy as np

def convert_and_flatten(y_true, y_pred, pos_label):
    y_true = np.where(y_true.flatten() == pos_label, 1, 0)
    y_pred = np.where(y_pred.flatten() == pos_label, 1, 0)

    return y_true, y_pred

def true_positives(y_true, y_pred, pos_label):
    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return np.sum(y_true * y_pred)

def false_positives(y_true, y_pred, pos_label):
    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return np.sum(np.where(y_pred - y_true == 1, 1, 0))

def false_negatives(y_true, y_pred, pos_label):
    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return np.sum(np.where(y_true - y_pred == 1, 1, 0))

# these functions are way faster than their sklearn counterparts

def precision(y_true, y_pred, pos_label):
    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)

    return tp / (tp + fp)

def recall(y_true, y_pred, pos_label):
    tp = true_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)

    return tp / (tp + fn)

def jaccard(y_true, y_pred, pos_label):
    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)

    return tp / (tp + fp + fn)

def f1(y_true, y_pred, pos_label):
    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)

    return tp / (tp + 0.5 * (fp + fn))
