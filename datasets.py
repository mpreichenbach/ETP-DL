from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import one_hot
from helper_functions import rotate


def dataset_gen(dim, batch_size, image_dir, mask_dir, rot8=True, v_flip=True, h_flip=False,
                oh_encode=False, n_labels=None, seed=1):

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
        x = one_hot(x, n_labels) if oh_encode else x

        return x

    image_datagen = ImageDataGenerator(rescale=1/255.0,
                                       horizontal_flip=h_flip,
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

    return zip_gen
