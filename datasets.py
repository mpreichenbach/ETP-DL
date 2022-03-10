from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helper_functions import rotate, label_to_oh
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS
import numpy as np


def dataset_gen(dim, batch_size, image_dir, mask_dir, rot8=True, v_flip=True, h_flip=False, seed=1):

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

    # add preprocessing steps here
    def preprocess(x):
        x = rotate(x) if rot8 else x

        return x

    image_datagen = ImageDataGenerator(horizontal_flip=h_flip,
                                       # if you use preprocessing layers in a model, you likely don't need to rescale
                                       # rescale=1/255.0,
                                       vertical_flip=v_flip,
                                       preprocessing_function=preprocess)

    mask_datagen = ImageDataGenerator(horizontal_flip=h_flip,
                                      vertical_flip=v_flip,
                                      preprocessing_function=preprocess)

    image_generator = image_datagen.flow_from_directory(image_dir,
                                                        target_size=(dim, dim),
                                                        color_mode='rgb',
                                                        class_mode=None,
                                                        batch_size=batch_size,
                                                        # save_to_dir="Data/images/",
                                                        shuffle=True,
                                                        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(mask_dir,
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


# this is a custom generator given at https://github.com/keras-team/keras/issues/3059, in case you need more control
# over preprocessing functions. For example, ImageDataGenerators don't allow one-hot encoding in preprocessing.

def train_generator(image_dir, mask_dir, batch_size, rot=True, v_flip=True, rescale=1):
    list_images = os.listdir(image_dir)
    np.random.shuffle(list_images)
    ids_train_split = range(len(list_images))
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                img = cv2.imread(os.path.join(image_dir, list_images[id]), cv2.IMREAD_COLOR)
                mask = cv2.imread(os.path.join(mask_dir, list_images[id]), cv2.IMREAD_GRAYSCALE)

                if rot:
                    k_rot = np.random.randint(0, 4)
                    img = np.rot90(img, k=k_rot)
                    mask = np.rot90(mask, k=k_rot)

                if v_flip and np.random.randint(0, 2) == 1:
                    img = np.flip(img, axis=0)
                    mask = np.flip(mask, axis=0)

                x_batch.append(img)
                y_batch.append(mask)

            # you likely don't need a rescale factor if you're using preprocessing layers in a model
            x_batch = np.array(x_batch, np.float32) * rescale
            y_batch = np.array(y_batch, np.float32)

            yield x_batch, y_batch
