import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from preprocess_helper import preprocess, process_path
import os
from unet import conv_block, upsampling_block, unet_model

IMAGE_DIR = "./data/CameraRGB"
MASK_DIR  = "./data/CameraMask"


files = sorted(os.listdir(IMAGE_DIR))  # image-mask matching
image_list = [os.path.join(IMAGE_DIR, f) for f in files]
mask_list  = [os.path.join(MASK_DIR,  f) for f in files]

# TensorFlow dataset
image_filenames = tf.constant(image_list)
mask_filenames  = tf.constant(mask_list)

dataset = tf.data.Dataset.from_tensor_slices(
    (image_filenames, mask_filenames)
)
image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)

img_height = 96
img_width = 128
num_channels = 3

unet = unet_model((img_height, img_width, num_channels))
unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
EPOCHS = 5
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
model_history = unet.fit(train_dataset, epochs=EPOCHS)

