import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
from SemSeg_classes import SemSeg
import tensorflow as tf

# these lines seem to fix memory errors
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# tensorflow now only sees GPUs 1,2,3 but calls them 0,1,2; the following line allows for forward passes on multiple GPUs
strategy = tf.distribute.MirroredStrategy(devices=["GPU:" + str(n) for n in [0, 1, 2]])
pretrained_weights = "VGG19"

with strategy.scope():
	cnn = SemSeg()
	cnn.initial_model(backbone=pretrained_weights)
	cnn.model = tf.keras.models.clone_model(cnn.model)
	cnn.model.compile(loss="categorical_crossentropy", optimizer="Adam")

cnn.load_generator(train_path="Inria/Train/", val_path="Inria/Validation/", batch_size=192)
cnn.train_model(epochs=200, monitor="loss", save_path="Saved Models/" + pretrained_weights + " Plus/")
