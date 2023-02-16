"""
SpecSegv2 testing for use with variable dimensions and videos
Training progress is recorded using W&B
W&B reports are used to review the results

# Uses Packages:
#     Python 3.8
#     CUDA 11.3
#     cuDnn 8.0
#     Tensorflow 2.5 + Keras 2.4
#     OmegaConf

# (C) 2021 Atif Anwer,
# Email: atif.anwer@u-bourgogne.fr
# -----------------------------------------------------------
"""
import glob
import itertools
import logging
import os
import pathlib
import random
import time
from datetime import datetime
from pickle import FALSE

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from keras import layers, mixed_precision
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal, RandomUniform
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dropout,
                          Input, Lambda, MaxPooling2D, UpSampling2D,
                          concatenate)
from keras.models import Model, load_model
from keras.optimizers import adam_v2 as Adam
from matplotlib import cm
from matplotlib import pyplot as plt
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from packaging import version
from PIL import Image
# from python.ops.init_ops_v2 import Initializer
from skimage.color import rgb2gray
from skimage.io import imread, imread_collection, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import normalize
from tqdm import tqdm

# keras = tf.keras


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# To debug @tf.functions in vscode (otherwise breakpoints dont work)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Configuring python logger
# Ref: https://www.youtube.com/watch?v=pxuXaaT1u3k
# Levels = Critical, Error, Warning, INFO, Debug
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")



# Decorrator for timing a function
# Ref: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
def _time(f):
    def wrapper (*args):
        start = time.time()
        r = f(*args)
        end = time.time()
        tqdm.write ("%s timed %f" % (f.__name__, end-start) )
        return r
    return wrapper


# ------------------------------------------------
#
# ███████ ██████  ███████  ██████ ███████ ███████  ██████
# ██      ██   ██ ██      ██      ██      ██      ██
# ███████ ██████  █████   ██      ███████ █████   ██   ███
#      ██ ██      ██      ██           ██ ██      ██    ██
# ███████ ██      ███████  ██████ ███████ ███████  ██████
#
# ------------------------------------------------

@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def specseg(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c1)
    c1 = BatchNormalization(axis=-1)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c2)
    c2 = BatchNormalization(axis=-1)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c3)
    c3 = BatchNormalization(axis=-1)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c4)
    c4 = BatchNormalization(axis=-1)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c5)
    c5 = BatchNormalization(axis=-1)(c5)

    #Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    opt = tf.keras.optimizers.Adam(learning_rate=0.00002)

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model.compile(optimizer=opt, loss=total_loss, metrics=metrics)
    model.summary()

    return model


# ------------------------------------------------
#
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
# ------------------------------------------------

# Adding hydra config file to load parameters at file load
@hydra.main(version_base = None, config_path = "conf", config_name = "config")
@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def SpecSegv2( cfg: DictConfig ) -> None:
    """Main function.
    1, Prints the current hydra configuration in use
    2. Loads SpecSeg model
    3. Load checkpoint
    4. Load dataset
    5. Process images
    """

    # ---- hydra config ----
    logging.info("Current Hydra config in use:")

    # debug print for hydra
    print(OmegaConf.to_yaml(cfg))


    # ---- Load SpecSeg Model ----
    # model = specseg(cfg.params.IMG_HEIGHT, cfg.params.IMG_WIDTH, cfg.params.IMG_CHANNELS)
    model = specseg(None, None, cfg.params.IMG_CHANNELS)

    model.load_weights('SpecSeg_weights.hdf5') #Trained for 50 epochs and then additional 100

    # ---- Load Dataset in eithre RAM or as TF dataset ----
    # Load dataset using the Tensorflow dataset loader or directly into RAM
    if cfg.params.dataset_load == "TFLoader":
        length_dataset, Dataset, test_datset = datasetLoad_tf( cfg )
    elif cfg.params.dataset_load == "RGB_RAM":
        image_dataset, mask_dataset = datasetload_ram( cfg )
    elif cfg.params.dataset_load == "RESIZED_RAM":
        image_dataset, mask_dataset = datasetload_ram_resized( cfg )

    # Make the iterator from the zipped datasets
    iterator = (zip(image_dataset, mask_dataset))

    # ---- Test on Random images ----
    # Enter number of images to test on
    num_images = 10
    # The starting index number from the images loaded
    start = 25
    index = 0
    img_no = 0
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16,32))
    fig.tight_layout()
    for i in tqdm(range(start, start+num_images), desc='Testing Image #', unit=' images'):

        test_img_number = i

        if cfg.params.dataset_load == 'RGB_RAM':
            """ For testing Original RGB images """
            test_img = cv2.cvtColor(image_dataset[test_img_number], cv2.COLOR_RGB2GRAY)
            segmented_image = predict_patches(model, test_img, cfg.params.patch_size)

        elif cfg.params.dataset_load == 'RESIZED_RAM':
            """ For testing resized images """
            test_img = image_dataset[test_img_number]
            test_img_norm=test_img[:,:,0][:,:,None]
            test_img_input=np.expand_dims(test_img_norm, 0)
            segmented_image = (model.predict(test_img_input)[0,:,:,0] )

        # --------------TEST IMAGE-----------------
        plt.subplot(num_images,3,index+1)
        plt.title('Test Image')
        plt.axis('off')
        plt.imshow(image_dataset[test_img_number])
        # name="./results_randomTestImages/img_" + str(i) + '.png'
        # plt.imsave(name, test_img[:,:,0], cmap='gray')

        # --------------GT -----------------
        ground_truth=mask_dataset[test_img_number]
        plt.subplot(num_images,3,index+2)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.imshow(ground_truth, cmap='gray')
        # name="./results/gt" + str(i) + '.png'

        # --------------PREDICTION-----------------
        plt.subplot(num_images,3,index+3)
        plt.title('Predicted Specular Highlights')
        plt.axis('off')
        plt.imshow(segmented_image, cmap='gray')
        # name="./results_randomTestImages/prediction_" + str(i) + '.png'
        # plt.imsave(name, prediction, cmap='gray')

        index = 3 * (img_no+1)
        img_no += 1

    # plt.imsave('/results/output2.jpg', fig)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig('randomTestImages'+timestr+'-random.png')   # save the figure to file
    print ("Done and Done!")

    return

@_time
def predict_patches (model, image, patch_size):
    """
    Predict a large image by patching it into 256x256 images
    """

    # If the image is smaller than 256x256, resize it first to adapt to SpecSeg
    if np.shape(image)[0] < 256 or np.shape(image)[1] < 256:
                    image = cv2.resize(image, (256, 256))

    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    filled_patch = False
    for i in range(0, image.shape[0], 256):   #Steps of 256
        for j in range(0, image.shape[1], 256):  #Steps of 256
            #print(i, j)
            single_patch = image[i:i+patch_size, j:j+patch_size]
            if np.shape(single_patch)[0] <256 or np.shape(single_patch)[1] <256:
                # Reshape the patch and fill with 0 if the patches are at the corner
                filled_patch = True
                dim1 = np.shape(single_patch)[0]
                dim2 = np.shape(single_patch)[1]
                single_patch = np.pad(single_patch, [(256-dim1, 0), (0,256-dim2)], mode='constant', constant_values=(0,0))
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            if filled_patch == True:
                # Crop the predicted image back to the original size
                single_patch_prediction = single_patch_prediction[0:dim1, 0:dim2]
                # reset the flag for the next patches
                filled_patch = False
                single_patch_shape = (dim1, dim2)
                segm_img[i:i+dim1, j:j+dim2] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
            else:
                segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])

            # print(f"Finished processing patch number {patch_num} at position {i},{j}")
            patch_num+=1
    return segm_img

# ------------------------------------------------
#
# ██████   █████  ████████  █████  ██       ██████   █████  ██████  ███████ ██████
# ██   ██ ██   ██    ██    ██   ██ ██      ██    ██ ██   ██ ██   ██ ██      ██   ██
# ██   ██ ███████    ██    ███████ ██      ██    ██ ███████ ██   ██ █████   ██████
# ██   ██ ██   ██    ██    ██   ██ ██      ██    ██ ██   ██ ██   ██ ██      ██   ██
# ██████  ██   ██    ██    ██   ██ ███████  ██████  ██   ██ ██████  ███████ ██   ██
#
# Initializing the dataset from folders
# ------------------------------------------------
@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def datasetLoad_tf( ):
    """
    # --------------------------------------------------------------
    #                   LOAD IMAGES AS TF DATASET                  |
    # --------------------------------------------------------------
    """
    rootfolder = cfg.data_dir
    valfolder  = cfg.val_dir
    # FOR SHMGAN Dataset
    path1 = os.path.join( rootfolder, 'HighlightImages' )
    path2 = os.path.join( rootfolder, 'HighlightMasks' )
    valpath   = os.path.join( valfolder, 'HighlightImages' )

    data_dir1 = pathlib.Path( path1 )
    data_dir2 = pathlib.Path( path2 )
    data_dir3 = pathlib.Path( valpath )

    # NOTE:  => The generated datasets do not have any lables

    rgb_images = tf.keras.preprocessing.image_dataset_from_directory(
            str( data_dir1 ),
                labels           = None,
                # label_mode       = 'categorical',
                color_mode       = 'rgb',
                validation_split = 0.2,
                subset           = "training",
                shuffle          = False,
                seed             = 1337,
                image_size       = (cfg.image_size, cfg.image_size),
                batch_size       = 1
            ) \
            .cache() \
            .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
            .map(lambda x: x if cfg.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
            .prefetch(25)
            # .map(lambda x: tf.image.per_image_standardization( x ) ) \
            # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
    # Manually update the labels (?)
    rgb_images.class_names = 'RGB'

    masks = tf.keras.preprocessing.image_dataset_from_directory(
            str( data_dir2 ),
                labels           = None,
                # label_mode       = 'categorical',
                color_mode       = 'rgb',
                validation_split = 0.2,
                subset           = "training",
                shuffle          = False,
                seed             = 1337,
                image_size       = (cfg.image_size, cfg.image_size),
                batch_size       = 1
            ) \
            .cache() \
            .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
            .map(lambda x: x if cfg.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
            .prefetch(25)
            # .map(lambda x: tf.image.per_image_standardization( x ) ) \
            # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
    masks.class_names = 'MASK'

    val_images = tf.keras.preprocessing.image_dataset_from_directory(
            str( data_dir1 ),
                labels           = None,
                # label_mode       = 'categorical',
                color_mode       = 'rgb',
                validation_split = 0.2,
                subset           = "validation",
                shuffle          = False,
                seed             = 1337,
                image_size       = (cfg.image_size, cfg.image_size),
                batch_size       = 1
            ) \
            .cache() \
            .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
            .map(lambda x: x if cfg.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
            .prefetch(25)
            # .map(lambda x: tf.image.per_image_standardization( x ) ) \
            # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
    # Manually update the labels (?)
    val_images.class_names = 'RGB Validation'

    # ZIP the datasets into one dataset
    loadedDataset = tf.data.Dataset.zip ( ( rgb_images, masks ) )
    test_datset = tf.data.Dataset.zip ( val_images )

    # dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)

    # Sauce: https://www.tensorflow.org/guide/data_performance_analysis#3_are_you_reaching_high_cpu_utilization
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    loadedDataset = loadedDataset.with_options(options)

    # -------------------------------------------------------
    # Repeat/parse the loaded dataset for the same number as epochs...
    # cahces and prefetches the datasets for performance
    # repeat for epochs
    # TODO: Check for performance
    loadedDataset = loadedDataset.cache().repeat( cfg.num_epochs ).prefetch( buffer_size =25)
    # -------------------------------------------------------

    # return the number of files loaded , to calculate iterations per batch
    length_dataset = len(np.concatenate([i for i in rgb_images], axis=0))
    # returns the zipped dataset for use with iterator
    return length_dataset, loadedDataset

@_time
def datasetload_ram( cfg: DictConfig ):
    """
    # --------------------------------------------------------------
    #                       LOAD IMAGES IN RAM                     |
    # --------------------------------------------------------------
    """
    image_directory = cfg.files.test_img_dir
    mask_directory  = cfg.files.test_mask_dir

    SIZE = 256
    image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.
    mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

    logging.info("Loading Test images")
    images = natsorted(os.listdir(image_directory))
    for i, image_name in enumerate(tqdm(images, desc='Test Images', unit=' images')):    #Remember enumerate method adds a counter and returns the enumerate object
        if (image_name.split('.')[1] == 'png'):
            # Loading images as RGB
            image = cv2.cvtColor( cv2.imread(image_directory+image_name), cv2.COLOR_BGR2RGB)
            # Load image as Grayscale
            # image = cv2.imread(image_directory+image_name, 0)
            image = Image.fromarray(image)
            # image = image.resize( (SIZE, SIZE) )
            # q = np.array(image)
            image_dataset.append( np.array(image) )

    logging.info("Loading Test Masks")
    masks = natsorted(os.listdir(mask_directory))
    for i, image_name in enumerate(tqdm(masks, desc='Test Masks', unit=' images')):
        if (image_name.split('.')[1] == 'png'):
            # Loading images as GREYSCALE mode
            image = cv2.imread(mask_directory+image_name, 0)
            image = Image.fromarray(image)
            # image = image.resize( (SIZE, SIZE) )
            mask_dataset.append( np.array(image) )

    #Normalize images
    # image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
    #D not normalize masks, just rescale to 0 to 1.
    # mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
    mask_dataset = np.array( mask_dataset, dtype=object )/255.0


    # X_train, X_test, y_train, Y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

    # cfg.IMG_HEIGHT = image_dataset.shape[1]
    # cfg.IMG_WIDTH  = image_dataset.shape[2]
    # cfg.IMG_CHANNELS = image_dataset.shape[3]

    # return X_train, y_train, X_test, Y_test
    return image_dataset, mask_dataset

@_time
def datasetload_ram_resized( cfg: DictConfig ):
    """
    # --------------------------------------------------------------
    #                       LOAD IMAGES IN RAM                     |
    # --------------------------------------------------------------
    """
    image_directory = cfg.files.test_img_dir
    mask_directory  = cfg.files.test_mask_dir

    SIZE = 256
    image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.
    mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

    logging.info("Loading Test images")
    images = natsorted(os.listdir(image_directory))
    for i, image_name in enumerate(tqdm(images, desc='Test Images', unit=' images')):    #Remember enumerate method adds a counter and returns the enumerate object
        if (image_name.split('.')[1] == 'png'):
            # Loading images as RGB
            # image = cv2.cvtColor( cv2.imread(image_directory+image_name), cv2.COLOR_BGR2RGB)
            # Load image as Grayscale
            image = cv2.imread(image_directory+image_name, 0)
            image = Image.fromarray(image)
            image = image.resize( (SIZE, SIZE) )
            # q = np.array(image)
            image_dataset.append( np.array(image) )

    logging.info("Loading Test Masks")
    masks = natsorted(os.listdir(mask_directory))
    for i, image_name in enumerate(tqdm(masks, desc='Test Masks', unit=' images')):
        if (image_name.split('.')[1] == 'png'):
            # Loading images as GREYSCALE mode
            image = cv2.imread(mask_directory+image_name, 0)
            image = Image.fromarray(image)
            image = image.resize( (SIZE, SIZE) )
            mask_dataset.append( np.array(image) )

    #Normalize images
    image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
    #D not normalize masks, just rescale to 0 to 1.
    # mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
    mask_dataset = np.array( mask_dataset )/255.0


    # X_train, X_test, y_train, Y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

    # cfg.IMG_HEIGHT = image_dataset.shape[1]
    # cfg.IMG_WIDTH  = image_dataset.shape[2]
    # cfg.IMG_CHANNELS = image_dataset.shape[3]

    # return X_train, y_train, X_test, Y_test
    return image_dataset, mask_dataset

# ------------------------------------------------
#               __  .__   __.  __  .___________.
#              |  | |  \ |  | |  | |           |
#              |  | |   \|  | |  | `---|  |----`
#              |  | |  . `  | |  |     |  |
#              |  | |  |\   | |  |     |  |
#  ______ _____|__| |__| \__| |__|     |__|     ______ ______
# |______|______|                              |______|______|
# ------------------------------------------------
if __name__ == "__main__":
    """
    Reduces Tensorflow messages other than errors or important messages
    '0' #default value, output all information
    '1' #Block notification information (INFO)
    '2' #Shield notification information and warning information (INFO\WARNING)
    '3' #Block notification messages, warning messages and error messages (INFO\WARNING\FATAL)
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    SpecSegv2()


# REFERENCES:
