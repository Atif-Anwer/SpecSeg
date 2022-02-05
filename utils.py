"""
# -----------------------------------------------------------
    VARIOUS UTILITIES AND PLOT FUNCTIOS USED IN SHMGAN

Uses Packages:
    Python 3.8
    CUDA 11.3
    cuDnn 8.0
    Tensorflow 2.5/2.6 + Keras 2.4

(C) 2021-22 Atif Anwer, INSA Rouen, France
Email: atif.anwer@insa-rouen.fr

BLOCKS:
    1. SHMGAN FUNCTIONS
    2. SAVE FUNCTIONS
    3. PRINT STUFF
    4. VARIOUS PLOT FUNCTIONS

# -----------------------------------------------------------
"""

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

from datetime import datetime
from tensorflow import keras


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------------------------------------------
#                       SHMGAN FUNCTIONS                       |
# --------------------------------------------------------------


def check_gpu():
    # ----------------------------------------
    # SETUP GPU
    # ----------------------------------------
    # # Testing and enabling GPU
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print(tf.test.is_built_with_cuda())
    gpus = tf.config.list_physical_devices('GPU')
    print("[ => ] Num GPUs Available: ", len(gpus))
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        os.system("nvidia-smi --query-gpu=gpu_name --format=csv")

    # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
    # tf.config.LogicalDeviceConfiguration(memory_limit=6124)

    # tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6124)])

    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.experimental.enable_tensor_float_32_execution(enabled=True)

    # https://www.tensorflow.org/api_docs/python/tf/config/optimizer/set_experimental_options
    tf.config.optimizer.set_experimental_options({'constant_folding': True})
    tf.config.optimizer.set_experimental_options({'layout_optimizer': True})
    tf.config.optimizer.set_experimental_options({'remapping': True})
    tf.config.optimizer.set_experimental_options({'loop_optimization': True})
    tf.config.optimizer.set_experimental_options(
        {'pin_to_host_optimization': True})
    tf.config.set_soft_device_placement(True)
    # https://stackoverflow.com/questions/65322700/tensorflow-keras-consider-either-turning-off-auto-sharding-or-switching-the-a
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    # os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
    # os.environ["TF_NUM_INTEROP_THREADS"] = "8"
    # os.environ["OMP_NUM_THREADS"] = "16"
    # os.environ["KMP_BLOCKTIME"] = "1"
    # os.environ["KMP_SETTINGS"] = "1"
    # os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    # enable mixed precision
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # # https://www.tensorflow.org/guide/profiler
    # os.environ['TF_GPU_THREAD_MODE']='gpu_private'
    # os.environ['TF_GPU_THREAD_COUNT']='1'
    # # Max out the L2 cache
    # _libcudart = ctypes.CDLL('libcudart.so')
    # # Set device limit on the current device
    # # cudaLimitMaxL2FetchGranularity = 0x05
    # pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    # _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    # _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    # assert pValue.contents.value == 128


# ------------------------------------------
def calculate_estimate_diffuse(args, OriginalImageStack, imgStack_0deg, imgStack_45deg, imgStack_90deg, imgStack_135deg):
    filepath = args.path
    dirname = os.path.dirname(filepath)
    # Estimated diffuse images pre-calculated and saved
    estDiffFolder = os.path.join(dirname, "Estimated_Diffuse")

    b = []
    g = []
    r = []
    estimated_diffuse_stack = []
    filenames_est_diffuse = []
    i = 0
    for orig, img0, img45, img90, img135 in zip(OriginalImageStack, imgStack_0deg, imgStack_45deg, imgStack_90deg, imgStack_135deg):
        # Note: Each img variable is a 3 channel image; so we can split it up in BGR
        blue, green, red = cv2.split(img0)
        b.append(blue)
        g.append(green)
        r.append(red)

        blue, green, red = cv2.split(img45)
        b.append(blue)
        g.append(green)
        r.append(red)

        blue, green, red = cv2.split(img90)
        b.append(blue)
        g.append(green)
        r.append(red)

        blue, green, red = cv2.split(img135)
        b.append(blue)
        g.append(green)
        r.append(red)

        b_min = np.amin(b, axis=0)
        g_min = np.amin(g, axis=0)
        r_min = np.amin(r, axis=0)

        merged = cv2.merge([b_min, g_min, r_min])
        i += 1

        # WRITE the image to a file if required. Can eb commented out if req
        name = estDiffFolder + "/" + 'Result_' + str(i) + '_ed.png'
        # cv2.imwrite(name, merged)
        filenames_est_diffuse.append(name)

        # Stack the estimated diffuse images for later use in loop
        estimated_diffuse_stack.append(merged)
        #
        # # DEBUG STACK DISPLAY
        # Horizontal1 = np.hstack([orig, merged])
        # # Debug display
        # cv2.namedWindow("Loaded Image", cv2.WINDOW_NORMAL)
        # # cv2.resizeWindow("dice", 600,600)
        # cv2.imshow("Loaded Image", Horizontal1)
        # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # # i += 1
        #
        # clear data before next loop; avoiding any data overwriting issues
        b.clear()
        g.clear()
        r.clear()
        merged.fill(0)  # clear the vars before calculating

        return estimated_diffuse_stack


# ------------------------------------------
# White balance
#  Source: https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
# ------------------------------------------
# def white_balance( input_image ):
#     result = cv2.cvtColor( input_image, cv2.COLOR_RGB2LAB )
#     avg_a = np.average( result[:, :, 1] )
#     avg_b = np.average( result[:, :, 2] )
#     result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
#     result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
#     whiteBalancedImage = cv2.cvtColor( result, cv2.COLOR_LAB2RGB )
#     return whiteBalancedImage


# ------------------------------------------------------------
#                       SAVE FUNCTIONS                       |
# ------------------------------------------------------------
def save_dataset_hdf5(image_stack):
    save_path = './estimated_diffuse_images.hdf5'
    hf = h5py.File(save_path, 'a')  # open a hdf5 file

    dset = hf.create_dataset('default', data=image_stack,
                             compression="gzip", compression_opts=9)
    hf.close()  # close the hdf5 file
    print('\n [ => ] Dataset Saved. hdf5 file size: %d bytes' %
          os.path.getsize(save_path))


# --------------------------------------------------------------
#                       PRINT STUFF                            |
# --------------------------------------------------------------
# ------------------------------------------
# Print iterations progress
#  https://github.com/Kal213/StarGAN-Tutorial-Tensorflow-2.3/blob/main/datagen.py
# Usage: printProgressBar(step % 1000, 999, decimals=2)
# ------------------------------------------
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r %s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
        print()

# --------------------------------------------------------------
#                       VARIOUS PLOT FUNCTIONS                 |
# --------------------------------------------------------------

# ------------------------------------------


# ------------------------------------------
# https://github.com/Ha0Tang/AttentionGAN/blob/68a478a944bb45d288d67f99fe110ddf087fd84d/AttentionGAN-v1-multi/solver.py#L123
# --------------------------------------------------
"""Convert the range from [-1, 1] to [0, 1]."""
# --------------------------------------------------


def rescale_01(input_tensor):
    # from [-1,1] to [0,1]
    # de_normalized = (input_tensor + 1.0) * 127.5 / 255.0

    # from [min, max] to [0,1]
    # normalize_value = (value − min_value) / (max_value − min_value)
    rescaled_01 = tf.math.divide_no_nan(
        tf.math.subtract(input_tensor, tf.math.reduce_min(input_tensor)),
        tf.math.subtract(tf.math.reduce_max(input_tensor), tf.math.reduce_min(input_tensor)))
    # de_normalized = tf.clip_by_value( de_normalized, clip_value_min=0, clip_value_max=1)
    return rescaled_01


# ------------------------------------------
# Plot an image grid
# Sauce: https://www.tensorflow.org/tensorboard/image_summaries
def image_grid(im1, im2, im3, im4, im5):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(15, 5))

    plt.subplot(1, 5, 1, title="0")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow((tf.squeeze(im1).numpy().astype("float32")))

    plt.subplot(1, 5, 2, title="45")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow((tf.squeeze(im2).numpy().astype("float32")))

    plt.subplot(1, 5, 3, title="90")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow((tf.squeeze(im3).numpy().astype("float32")))

    plt.subplot(1, 5, 4, title="135")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow((tf.squeeze(im4).numpy().astype("float32")))

    plt.subplot(1, 5, 5, title="ED")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow((tf.squeeze(im5).numpy().astype("float32")))

    return figure

# ------------------------------------------

# Taking an input tensor and plotting each channel (image+mask) for debugging randomness


def debug_plot(input_tensor):
    figure = plt.figure(figsize=(15, 10))
    input_tensor = tf.squeeze(input_tensor)
    channel = 0

    for index in range(5):
        channel += 1
        figure.add_subplot(2, 5, channel, title=str(channel-1))
        # plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow((input_tensor[:, :, channel-1]
                    ).numpy().astype("float32"), cmap=cm.gray, )

        figure.add_subplot(2, 5, channel+5, title="Mask "+str(channel+5))
        # plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(input_tensor[:, :, channel-1+5],
                   cmap=cm.gray, vmin=0, vmax=1)

    # plt.show()
    return figure

# ------------------------------------------
# Universal debug plotting for single or multi-channel images. Call whenever required


def plot_single_image(input_tensor, title=""):
    figure = plt.figure(figsize=(10, 15))
    if len(input_tensor[0, 0, 0, :]) == 1:
        plt.imshow(
            (tf.squeeze(input_tensor).numpy().astype("float32")),  cmap=cm.gray)
        plt.title(label=title)
    else:
        # qq = tf.squeeze( input_tensor )
        # figure.add_subplot( 4, 1, 1, title="Orig")
        # plt.imshow( ( qq ).numpy().astype("float32") )

        # y,cb,cr = tf.split( qq, 3, axis=2)
        # figure.add_subplot( 4, 1, 2, title="Ch1")
        # plt.imshow(y)
        # figure.add_subplot( 4, 1, 3, title="Ch2")
        # plt.imshow(cb)
        # figure.add_subplot( 4, 1, 4, title="Ch3")
        # plt.imshow(cr)

        figure.add_subplot(4, 1, 1, title="Orig")
        plt.imshow(tf.squeeze(input_tensor).numpy().astype("float32"))

        y, cb, cr = tf.split(tf.squeeze(input_tensor), 3, axis=2)
        figure.add_subplot(4, 1, 2, title="Ch1")
        plt.imshow(tf.squeeze(rescale_01(y)).numpy().astype(
            "float32"),  cmap=cm.gray)
        figure.add_subplot(4, 1, 3, title="Ch2")
        plt.imshow(tf.squeeze(rescale_01(cb)).numpy().astype(
            "float32"),  cmap=cm.gray)
        figure.add_subplot(4, 1, 4, title="Ch3")
        plt.imshow(tf.squeeze(rescale_01(cr)).numpy().astype(
            "float32"),  cmap=cm.gray)


# ------------------------------------------
# Given a set of images, show an animation.
# animate(np.stack(images))
# Source: https://www.tensorflow.org/hub/tutorials/tf_hub_generative_image_module
# def animate(images):
#     images = np.array(images)
#     converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
#     imageio.mimsave('./animation.gif', converted_images)
#     return embed.embed_file('./animation.gif')


# ------------------------------------------
# Source: https://www.tensorflow.org/tensorboard/image_summaries
# Displaying images in Tensorboard Grids
# ------------------------------------------
# def plot_to_image(figure):
#     """Converts the matplotlib plot specified by 'figure' to a PNG image and
#     returns it. The supplied figure is closed and inaccessible after this call."""
#     # Save the plot to a PNG in memory.
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     # Closing the figure prevents it from being displayed directly inside
#     # the notebook.
#     plt.close(figure)
#     buf.seek(0)
#     # Convert PNG buffer to TF image
#     image = tf.image.decode_png(buf.getvalue(), channels=4)
#     # Add the batch dimension
#     image = tf.expand_dims(image, 0)
#     return image
