import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf

def load_data(path):
    images = sorted(glob(os.path.join(path, "Train_RGB_256/*")))
    masks = sorted(glob(os.path.join(path, "Train_Labels_256/*")))

    return images, masks

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x /= 255.0
    x = x.astype(np.float32)

    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x /= 255.0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)

    return x

def preprocess(x, y):
    def f(x, y):
        # are these lines necessary?
        # x = x.decode()
        # y = y.decode()

        x = read_image(x)
        y = read_image(y)

        return x, y

    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([256, 256, 3])
    masks.set_shape([256, 256, 1])

    return images, masks

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buff_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)

    return dataset

if __name__ == "__main__":
    path = "Data/Potsdam/Numpy Arrays/Train/"
    images, masks = load_data(path)
    print(f"Images: {len(images)} - Masks: {len(masks)}")

    dataset = tf_dataset(images, masks)
    counter = 0
    for x, y in dataset:
        x = x.numpy()
        y = y.numpy()

        cv2.imwrite("Data/Potsdam/Numpy Arrays/test/images/" + str(counter) + ".png")

        y = np.squeeze(y, axis=-1)
        cv2.imwrite("Data/Potsdam/Numpy Arrays/test//masks/" + str(counter) + ".png")
        counter += 1

        break

