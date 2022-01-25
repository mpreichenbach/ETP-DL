from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helper_functions import rotate


def dataset_gen(dim, image_dir, mask_dir, rotate=True, v_flip=True, h_flip=False, seed=1):
    """Creates a pair of iterators which will transform the images/masks in identical ways.

    Args:
        dim (int): the width of an image tile,
        image_dir (str): the path to the folder containing the folder of images,
        mask_dir (str): the path to the folder containing the folder of masks,
        rotate (bool): whether to perform a random 0, 90, 180, or 270 degree rotation,
        v_flip (bool): whether to perform a flip over the vertical axis,
        h_flip (bool): whether to perform a flip over the vertical axis (unnecessary when rotation, v_flip=True,
        seed (int): the random seed to set for transformations."""

    if rotate:
        rot8 = rotate

    image_datagen = ImageDataGenerator(rescale=1/255.0,
                                       horizontal_flip=h_flip,
                                       vertical_flip=v_flip,
                                       preprocessing_function=rot8)

    mask_datagen = ImageDataGenerator(horizontal_flip=h_flip,
                                      )



# we create two instances with the same arguments
data_gen_args = dict(preprocessing_function=rotate)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

image_generator = image_datagen.flow_from_directory(
    'D:/ETP Data/Potsdam/Numpy Arrays/flow from/images/',
    target_size=(512, 512),
    class_mode=None,
    batch_size=1,
    shuffle=False,
    save_to_dir='D:/ETP Data/Potsdam/Numpy Arrays/flow to/images/',
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    'D:/ETP Data/Potsdam/Numpy Arrays/flow from/masks/',
    target_size=(512, 512),
    class_mode=None,
    batch_size=1,
    shuffle=False,
    save_to_dir='D:/ETP Data/Potsdam/Numpy Arrays/flow to/masks/',
    seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)


