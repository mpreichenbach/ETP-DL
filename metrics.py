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

# enc = label_to_oh(label_holder, 6)
# np.save('Data/Potsdam/Downsampled/enc_10cm_256.npy', enc)
#
# for res in [50, 100, 200]:
#     for tile_dim in [256, 512]:
#         path = 'Data/Potsdam/Downsampled RGB/' + str(res) + 'cm/'
#         files = os.listdir(path)
#         n = len(files)
#         test = Image.open(path + files[0])
#         arr = np.asarray(test, dtype=np.uint8)
#         dim = arr.shape[0]
#         holder_list = []
#
#         for file in files:
#             print(file)
#             with Image.open(path + file) as im:
#                 arr = np.asarray(im, dtype=np.uint8)
#                 for i in range(int(dim / tile_dim)):
#                     for j in range(int(dim / tile_dim)):
#                         tile = arr[tile_dim * i: tile_dim * (i + 1), tile_dim * j: tile_dim * (j + 1)]
#                         tile = tile.reshape((1,) + tile.shape)
#                         holder_list.append(tile)
#
#         rgb = np.concatenate(holder_list, axis=0)
#
#         path = 'Data/Potsdam/Downsampled Labels/' + str(res) + 'cm/'
#         files = os.listdir(path)
#         n = len(files)
#         test = Image.open(path + files[0])
#         arr = np.asarray(test, dtype=np.uint8)
#         dim = arr.shape[0]
#         holder_list = []
#
#         for file in files:
#             print(file)
#             with Image.open(path + file) as im:
#                 arr = np.asarray(im, dtype=np.uint8)
#                 for i in range(int(dim / tile_dim)):
#                     for j in range(int(dim / tile_dim)):
#                         tile = arr[tile_dim * i: tile_dim * (i + 1), tile_dim * j: tile_dim * (j + 1)]
#                         tile = tile.reshape((1,) + tile.shape)
#                         holder_list.append(tile)
#
#         labels = np.concatenate(holder_list, axis=0)
#
#         list_to_drop = []
#         for n in range(labels.shape[0]):
#             if n % 50 == 0:
#                 print(n)
#             for i in range(labels.shape[1]):
#                 for j in range(labels.shape[2]):
#                     if tuple(labels[n, i, j]) not in rgb_list:
#                         list_to_drop.append(n)
#
#         rgb = np.delete(rgb, list_to_drop, axis=0)
#         labels = np.delete(rgb, list_to_drop, axis=0)
#
#         enc = label_to_oh(labels, 6)
#
#         np.save('Data/Potsdam/Numpy Arrays/Downsampled/rgb_' + str(res) + 'cm_' + str(tile_dim) + '.npy', rgb)
#         np.save('Data/Potsdam/Numpy Arrays/Downsampled/labels_' + str(res) + 'cm_' + str(tile_dim) + '.npy', labels)
#         np.save('Data/Potsdam/Numpy Arrays/Downsampled/enc_' + str(res) + 'cm_' + str(tile_dim) + '.npy', enc)



