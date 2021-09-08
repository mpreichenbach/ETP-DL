#####
# The following code creates a dataframe with various metrics for all saved models in a folder

potsdam = SemSeg(256)
s_test, m_test, e_test = potsdam.load_data(test_only=True)

metrics = pd.DataFrame(0, index=['VGG16_fine_tuned', 'VGG16_cc_then_iou', 'VGG19_fine_tuned', 'VGG19_cc_then_iou'],
                       columns=['Total Acc', 'IoU', 'Dice', 'GPU Inf Time', 'CPU Inf Time'])


# for cpu inference times, incorporate this code after importing tensorflow:
tf.config.set_visible_devices([], 'GPU')
# tf.debugging.set_log_device_placement(True)
#
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
#
# print(c)
#
date = '2021-08-18'
holder = np.zeros([4, 5, 256, 256, 3]).astype(np.uint8)
# choices = np.arange(0, 5)
choices = np.random.choice(len(s_test), size=5, replace=False)

for i in range(len(metrics.index)):
    name = metrics.index[i]
    p_name = name + '/'
    path = os.path.join('Saved Models', date, p_name)

    short_str = name[0:5]
    model = pt_model(short_str, (256, 256, 3), 6)
    model.load_weights(path)

    y_true = e_test[0:300]

    tic = time.perf_counter()
    y_pred = model.predict(s_test[0:300])
    toc = time.perf_counter()
    inf_time = (toc - tic) / 3

    acc = total_acc(y_true[0:300], y_pred[0:300])
    iou = iou_loss(y_true[0:300], y_pred[0:300]).numpy() / 3
    dice = iou_loss(y_true[0:300], y_pred[0:300]).numpy() / 3

    metrics.loc[name, 'Total Acc'] = round(float(acc), 2)
    metrics.loc[name, 'IoU'] = round(1 - iou, 2)
    metrics.loc[name, 'Dice'] = round(1 - dice, 2)
    # metrics.loc[name, 'GPU Inf Time'] = round(inf_time, 2)

    # uncomment the following to get CPU speeds
    metrics.loc[name, 'CPU Inf Time'] = round(inf_time, 2)

    for j in range(5):
        holder[i, j] = oh_to_rgb(vec_to_oh(model.predict(s_test[choices[j]].reshape((1, 256, 256, 3)))),
                                 potsdam.class_df).reshape((256, 256, 3))

    del model

fig, axs = plt.subplots(5, 6)
plt.rcParams.update({'font.size': 10})

for i in range(5):
    if i == 0:
        axs[i, 0].imshow(s_test[choices[i]])
        axs[i, 0].set_title("RGB")
        axs[i, 1].imshow(m_test[choices[i]])
        axs[i, 1].set_title("Ground Truth")
        axs[i, 2].imshow(holder[0, i])
        axs[i, 2].set_title('VGG16 FT')
        axs[i, 3].imshow(holder[1, i])
        axs[i, 3].set_title('VGG16 CC then IoU')
        axs[i, 4].imshow(holder[2, i])
        axs[i, 4].set_title('VGG19 FT')
        axs[i, 5].imshow(holder[3, i])
        axs[i, 5].set_title('VGG16 CC then IoU')

    else:
        axs[i, 0].imshow(s_test[choices[i]])
        axs[i, 1].imshow(m_test[choices[i]])
        axs[i, 2].imshow(holder[0, i])
        axs[i, 3].imshow(holder[1, i])
        axs[i, 4].imshow(holder[2, i])
        axs[i, 5].imshow(holder[3, i])


plt.setp(axs, xticks=[], yticks=[])
plt.tight_layout()
plt.show()