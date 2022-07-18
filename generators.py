import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helper_functions import label_to_oh
import numpy as np


def data_generator(image_dir,
                   mask_dir,
                   batch_size,
                   one_hot=False,
                   classes=None,
                   v_flip=True,
                   rot=True,
                   scale=1/255):
    """This is a custom generator, adapted from an example in a comment at
    https://github.com/keras-team/keras/issues/3059. By editing this function, you can include any preprocessing you
    want. For example, the Keras ImageDataGenerator class does not allow one-hot encoding of labeled imagery, but the
    generator created by this function can perform that.

    Args:
        image_dir (str): path to input imagery;
        mask_dir (str): path to labeled imagery;
        batch_size (int): the number of images to include in a batch;
        one_hot (bool): whether to perform a one-hot encoding on the labeled images;
        classes (int): the number of classes in labeled imagery, only needed if one_hot=True;
        rot (bool): whether to randomly rotate loaded images;
        v_flip (bool): whether to randomly flip images over the vertical axis;
        scale (float): the value to multiply input pixels by."""

    if one_hot and classes is None:
        raise Exception("Number of classes must be specified when performing one-hot encoding.")

    list_images = os.listdir(image_dir)
    np.random.shuffle(list_images)
    ids_train_split = range(len(list_images))
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]

            for ID in ids_train_batch:
                img = cv2.imread(os.path.join(image_dir, list_images[ID]), cv2.IMREAD_COLOR)
                mask = cv2.imread(os.path.join(mask_dir, list_images[ID]), cv2.IMREAD_GRAYSCALE)

                if rot:
                    k_rot = np.random.randint(0, 4)
                    img = np.rot90(img, k=k_rot)
                    mask = np.rot90(mask, k=k_rot)

                if v_flip and np.random.randint(0, 2) == 1:
                    img = np.flip(img, axis=0)
                    mask = np.flip(mask, axis=0)

                if one_hot:
                    mask = label_to_oh(mask, classes)

                img = img * scale

                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            yield x_batch, y_batch

# this does not allow many preprocessing functions (specifically, no one-hot encodings)
def keras_dataset_gen(dim, batch_size, rgb_path, mask_path, rot8=True, v_flip=True, h_flip=False, seed=1):

    """Creates a pair of iterators which will transform the images/masks in identical ways.

    Args:
        dim (int): the width of an image tile,
        classes (int): the number of labeled classes in the masks,
        batch_size (int): the number of images to batch together,
        image_dir (str): the path to the folder containing the folder of images,
        mask_dir (str): the path to the folder containing the folder of masks,
        rotate (bool): whether to perform a random 0, 90, 180, or 270 degree rotation,
        v_flip (bool): whether to perform a flip over the vertical axis,
        h_flip (bool): whether to perform a flip over the vertical axis (unnecessary when rotation, v_flip=True,
        seed (int): the random seed to set for transformations."""

    image_datagen = ImageDataGenerator(horizontal_flip=h_flip,
                                       # if you use preprocessing layers in a model, you likely don't need to rescale
                                       # rescale=1/255.0,
                                       vertical_flip=v_flip)

    mask_datagen = ImageDataGenerator(horizontal_flip=h_flip,
                                      vertical_flip=v_flip)

    image_generator = image_datagen.flow_from_directory(rgb_path,
                                                        target_size=(dim, dim),
                                                        color_mode='rgb',
                                                        class_mode=None,
                                                        batch_size=batch_size,
                                                        # save_to_dir="Data/images/",
                                                        shuffle=True,
                                                        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(mask_path,
                                                      target_size=(dim, dim),
                                                      color_mode='grayscale',
                                                      class_mode=None,
                                                      batch_size=batch_size,
                                                      # save_to_dir="Data/masks/",
                                                      shuffle=True,
                                                      seed=seed)

    zip_gen = zip(image_generator, mask_generator)

    for x, y in zip_gen:
        yield x, y
