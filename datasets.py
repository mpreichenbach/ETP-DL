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
    x = x.astype(np.float32)
    x /= 255.0

    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)

    return x

def preprocess(x, y):
    def f(x, y):
        # this function loads the individual images
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)

        return x, y

    # this performs a random flip of the images; 0 is up-down, 1 is left-right, 2 is neither
    f_int = tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32)
    if f_int == 1:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)

    # this performs a random rotation of the images
    r_int = tf.random.uniform([1], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k=r_int)
    y = tf.image.rot90(y, k=r_int)

    # this performs a random contrast adjustment on the RGB image
    # c_int = tf.random.uniform([1])
    # x = tf.image.adjust_contrast(x, contrast_factor=c_int)

    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([256, 256, 3])
    masks.set_shape([256, 256, 1])

    return images, masks

def tf_dataset(x, y, batch=16):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)

    return dataset

if __name__ == "__main__":
    path = "Data/Potsdam/Numpy Arrays/Train/"
    images, masks = load_data(path)
    images = images[0:32]
    masks = masks[0:32]
    print(f"Images: {len(images)} - Masks: {len(masks)}")

    dataset = tf_dataset(images, masks)
    counter = 0
    for x, y in dataset:
        x = x[0].numpy()
        y = y[0].numpy()
        x *= 255.0
        y *= 255.0
        x = x.astype(np.uint8)
        y = y.astype(np.uint8)
        print(x.shape)
        print(y.shape)

        cv2.imwrite("Data/Potsdam/Numpy Arrays/this is a test folder/images/" + str(counter) + ".png", x)

        y = np.squeeze(y, axis=-1)
        cv2.imwrite("Data/Potsdam/Numpy Arrays/this is a test folder/masks/" + str(counter) + ".png", y)
        counter += 1

        break

