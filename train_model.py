import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
from SemSeg_classes import SemSeg
import tensorflow as tf

# these lines seem to fix memory errors
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# tensorflow now only sees GPUs 2,3 but calls them 0,1; the following line allows for forward passes on multiple GPUs
strategy = tf.distribute.MirroredStrategy(devices=["GPU:" + str(n) for n in [0, 1]])

with strategy.scope():
	cnn = SemSeg()
	cnn.initial_model()
	cnn.model = tf.keras.models.clone_model(cnn.model)
	cnn.model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam")

cnn.load_generator(train_path="Inria/Train/", val_path="Inria/Validation/", batch_size=120)
cnn.train_model(epochs=200, monitor="val_loss", save_path="Saved Models/VGG19 No PT (weights)/")
