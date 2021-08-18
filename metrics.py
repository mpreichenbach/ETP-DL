import tensorflow as tf
from SemSeg_Classes import SemSeg

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

#####
# The following code creates a dataframe with various metrics for all saved models in a folder

potsdam = SemSeg(256)
s_test, m_test, e_test = potsdam.load_data(test_only=True)

metrics = pd.DataFrame(0, index=['VGG16_cc', 'VGG16_iou', 'VGG16_dice', 'VGG16_cc_iou', 'VGG19_cc', 'VGG19_iou',
                                 'VGG19_dice', 'VGG19_cc_iou'],
                       columns=['Total Acc', 'IoU', 'Dice', 'GPU Inf Time', 'CPU Inf Time'])

date = '2021-08-11'
holder = np.zeros([8, 5, 256, 256, 3]).astype(np.uint8)

for i in range(len(metrics.index)):
    name = metrics.index[i]
    path = os.path.join('Saved Models', date, name, '/')

    short_str = name[0:5]
    model = pt_model(short_str, (256, 256, 3), 6)
    model.load_weights(path)

    y_true = e_test[0:300]

    tic = time.perf_counter()
    y_pred = model.predict(s_test[0:300])
    toc = time.perf_counter()
    inf_time = toc - tic

    acc = np.mean([total_acc(y_true[0:100], y_pred[0:100]), total_acc(y_true[100:200], y_pred[100:200]),
                  total_acc(y_true[200:300], y_pred[200:300])])
    iou = 1 - np.mean([iou_loss(y_true[0:100], y_pred[0:100]), iou_loss(y_true[100:200], y_pred[100:200]),
                      iou_loss(y_true[200:300], y_pred[200:300])])
    dice = 1 - np.mean([dice_loss(y_true[0:100], y_pred[0:100]), dice_loss(y_true[100:200], y_pred[100:200]),
                       dice_loss(y_true[200:300], y_pred[200:300])])


    metrics.loc[name, 'Total Acc'] = round(acc, 2)
    metrics.loc[name, 'IoU'] = round(1 - iou, 2)
    metrics.loc[name, 'Dice'] = round(1 - dice, 2)
    metrics.loc[name, 'GPU Inf Time'] = round(inf_time, 2)

    for j in range(5):
        holder[i, j] = oh_to_rgb(vec_to_oh(model.predict(s_test[j].reshape((1, 256, 256, 3)))),
                                 potsdam.class_df).reshape((256, 256, 3))

    del model

#
#     # for cpu inference times, incorporate this code right after importing tensorflow:
#     #tf.debugging.set_log_device_placement(True)
#

# fig, axs = plt.subplots(5, 10)
# plt.rcParams.update({'font.size': 12})
#
# for i in range(5):
#     if i == 0:
#         axs[i, 0].imshow(s_test[i])
#         axs[i, 0].set_title("RGB")
#         axs[i, 1].imshow(m_test[i])
#         axs[i, 1].set_title("Ground Truth")
#         axs[i, 2].imshow(holder[0, i])
#         axs[i, 2].set_title('VGG16 CC')
#         axs[i, 3].imshow(holder[1, i])
#         axs[i, 3].set_title('VGG16 IoU')
#         axs[i, 4].imshow(holder[2, i])
#         axs[i, 4].set_title('VGG16 Dice')
#         axs[i, 5].imshow(holder[3, i])
#         axs[i, 5].set_title('VGG16 CC+IoU')
#         axs[i, 6].imshow(holder[4, i])
#         axs[i, 6].set_title('VGG19')
#         axs[i, 7].imshow(holder[5, i])
#         axs[i, 7].set_title('VGG19 IoU')
#         axs[i, 8].imshow(holder[6, i])
#         axs[i, 8].set_title('VGG19 Dice')
#         axs[i, 9].imshow(holder[7, i])
#         axs[i, 9].set_title('VGG19 CC+IoU')
#
#
#     else:
#         axs[i, 0].imshow(s_test[i])
#         axs[i, 1].imshow(m_test[i])
#         axs[i, 2].imshow(holder[0, i])
#         axs[i, 3].imshow(holder[1, i])
#         axs[i, 4].imshow(holder[2, i])
#         axs[i, 5].imshow(holder[3, i])
#         axs[i, 6].imshow(holder[4, i])
#         axs[i, 7].imshow(holder[5, i])
#         axs[i, 8].imshow(holder[6, i])
#         axs[i, 9].imsohw(holder[7, i])
#
# plt.setp(axs, xticks=[], yticks=[])
# plt.tight_layout()
# plt.show()
