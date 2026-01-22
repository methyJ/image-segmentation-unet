import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from test_utils import summary, comparator
from preprocess import preprocess, process_path
import os
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from unet import conv_block, upsampling_block

IMAGE_DIR = "./data/CameraRGB"
MASK_DIR  = "./data/CameraMask"

# ---- file pairing (same filename) ----
files = sorted(os.listdir(IMAGE_DIR))  # sorted: image-mask eşleşmesi garanti olsun
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

