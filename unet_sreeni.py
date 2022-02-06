"""
@author: Sreenivas Bhattiprolu
"""


from skimage.io import imread, imshow, imread_collection
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import numpy as np
import cv2
import tensorflow as tf
import os
import h5py
import io
import itertools
import tensorflow as tf
import ctypes
from matplotlib import cm, pyplot as plt
from packaging import version
from tensorflow.keras import mixed_precision
import random
from tensorflow.python.ops.init_ops_v2 import Initializer
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import time
import datetime
import pathlib
import gc
from tqdm import tqdm

from datetime import datetime
from tensorflow import keras


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------------------------------------------
#                       SHMGAN FUNCTIONS                       |
# --------------------------------------------------------------


def unet_sreeni( args ):

    seed = 42
    np.random.seed = seed

    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    TRAIN_PATH = '/home/atif/Documents/Datasets/WHU-specular-dataset/train/'
    TEST_PATH = '/home/atif/Documents/Datasets/WHU-specular-dataset/test/'

    path, dirs, train_ids = next(os.walk(TRAIN_PATH + '/HighlightImages'))
    path, dirs, mask_ids = next(os.walk(TRAIN_PATH + '/HighlightMasks'))

    train_ids.sort()
    mask_ids.sort()

    # train_ids = next(os.walk(TRAIN_PATH))[1]
    # test_ids = next(os.walk(TEST_PATH))[1]


    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    print('Resizing training images and masks')
    # for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    #     path = TRAIN_PATH
    #     img = imread(path +'HighlightImages/'+ id_)[:,:,:3]
    #     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #     X_train[n] = img  #Fill empty X_train with values from img
    #     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    #     mask = rgb2gray(imread(path +'HighlightMasks/'+ id_))
    #     mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=2)

    #     Y_train[n] = mask

    # test images
    # X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    # sizes_test = []
    # print('Resizing test images')
    # for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    #     path = TEST_PATH + id_
    #     img = imread(path + '/HighlightImages/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    #     sizes_test.append([img.shape[0], img.shape[1]])
    #     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #     X_test[n] = img

    # print('Done!')

    # image_x = random.randint(0, len(train_ids))
    # imshow(X_train[image_x])
    # plt.show()
    # imshow(np.squeeze(Y_train[image_x]))
    # plt.show()

    length_trainset, length_testset, rgb_train, masks_train, rgb_test = datasetLoad()

    # plt.figure(figsize=(10, 10))
    # for images in rgb_train.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("float32"))
    #         # plt.title(i)
    #         plt.axis("off")

    # plt.figure(figsize=(10, 10))
    # for images in masks_train.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         # plt.title(i)
    #         plt.axis("off")

    # plt.figure(figsize=(10, 10))
    # for images in rgb_test.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("float32"))
    #         # plt.title(i)
    #         plt.axis("off")

    #Build the model
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    ################################
    #Modelcheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint('SpecSeg.h5', verbose=1, save_best_only=True)

    callbacks = [ tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]

    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

    ####################################

    idx = random.randint(0, len(X_train))


    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    # preds_test = model.predict(X_test, verbose=1)


    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    # preds_test_t = (preds_test > 0.5).astype(np.uint8)


    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t))
    imshow(X_train[ix])
    plt.show()
    imshow(np.squeeze(Y_train[ix]))
    plt.show()
    imshow(np.squeeze(preds_train_t[ix]))
    plt.show()

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    imshow(X_train[int(X_train.shape[0]*0.9):][ix])
    plt.show()
    imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()



def datasetLoad( ):

    trainfolder = '/home/atif/Documents/Datasets/WHU-specular-dataset/train/'
    testfolder  = '/home/atif/Documents/Datasets/WHU-specular-dataset/test/'
    # FOR SHMGAN Dataset
    train_images = pathlib.Path( os.path.join( trainfolder, 'HighlightImages' ) )
    train_masks  = pathlib.Path( os.path.join( trainfolder, 'HighlightMasks' ) )
    test_images  = pathlib.Path( os.path.join( testfolder, 'HighlightImages' ) )

    num_epochs = 20
    image_size = 128

    # NOTE:  => The generated datasets do not have any lables

    rgb_train = tf.keras.preprocessing.image_dataset_from_directory(
            str( train_images ),
                labels           = None,
                # label_mode       = 'categorical',
                color_mode       = 'rgb',
                # validation_split = 0.2,
                # subset           = "training",
                shuffle          = False,
                seed             = 1337,
                image_size       = (image_size, image_size),
                batch_size       = 32
            ) \
            .cache() \
            .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
            .prefetch(25)
            # .map(lambda x: x if random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
            # .map(lambda x: tf.image.per_image_standardization( x ) ) \
            # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
    # Manually update the labels (?)
    rgb_train.class_names = 'RGB'

    masks_train = tf.keras.preprocessing.image_dataset_from_directory(
            str( train_masks ),
                labels           = None,
                # label_mode       = 'categorical',
                color_mode       = 'grayscale',
                # validation_split = 0.2,
                # subset           = "training",
                shuffle          = False,
                seed             = 1337,
                image_size       = (image_size, image_size),
                batch_size       = 32
            ) \
            .cache() \
            .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
            .prefetch(25)
            # .map(lambda x: x if random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
            # .map(lambda x: tf.image.per_image_standardization( x ) ) \
            # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
    masks_train.class_names = 'MASK'

    rgb_test = tf.keras.preprocessing.image_dataset_from_directory(
            str( test_images ),
                labels           = None,
                # label_mode       = 'categorical',
                color_mode       = 'rgb',
                validation_split = 0.2,
                subset           = "validation",
                shuffle          = False,
                seed             = 1337,
                image_size       = (image_size, image_size),
                batch_size       = 32
            ) \
            .cache() \
            .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
            .prefetch(25)
            # .map(lambda x: x if random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
            # .map(lambda x: tf.image.per_image_standardization( x ) ) \
            # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
    # Manually update the labels (?)
    rgb_test.class_names = 'RGB Validation'

    # # ZIP the datasets into one dataset
    # loadedDataset = tf.data.Dataset.zip ( ( rgb_train, masks_train ) )
    # test_datset = tf.data.Dataset.zip ( val_images )

    # dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)

    # Sauce: https://www.tensorflow.org/guide/data_performance_analysis#3_are_you_reaching_high_cpu_utilization
    # options = tf.data.Options()
    # options.experimental_threading.max_intra_op_parallelism = 1
    # loadedDataset = loadedDataset.with_options(options)


    # -------------------------------------------------------
    # Repeat/parse the loaded dataset for the same number as epochs...
    # cahces and prefetches the datasets for performance
    # repeat for epochs
    # TODO: Check for performance
    rgb_train = rgb_train.cache().repeat( num_epochs ).prefetch( buffer_size = 32)
    masks_train = masks_train.cache().repeat( num_epochs ).prefetch( buffer_size = 32)
    rgb_test = rgb_test.cache().repeat( num_epochs ).prefetch( buffer_size = 32)
    # -------------------------------------------------------

    # return the number of files loaded , to calculate iterations per batch
    length_trainset = len(np.concatenate([i for i in rgb_train], axis=0))
    length_testset = len(np.concatenate([i for i in rgb_train], axis=0))
    # returns the zipped dataset for use with iterator
    return length_trainset, length_testset, rgb_train, masks_train, rgb_test