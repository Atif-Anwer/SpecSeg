"""
# -----------------------------------------------------------
SHMGAN -  Detection of Specular Highlights

Uses Packages:
    Python 3.8
    CUDA 11.3
    cuDnn 8.0
    Tensorflow 2.5/2.6 + Keras 2.4

(C) 2021-22 Atif Anwer, INSA Rouen, France
Email: atif.anwer@insa-rouen.fr

# -----------------------------------------------------------
"""
# import comet_ml at the top of your file
from comet_ml import Experiment

import numpy as np
import os
# import cv2
# import random
# from skimage.color.colorconv import ycbcr2rgb
from tensorflow.python.ops.init_ops_v2 import Initializer
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import time
import datetime
import pathlib
import gc

# from PIL import Image
from functools import partial
from utils import check_gpu, printProgressBar, rescale_01, image_grid, debug_plot, plot_single_image
from tensorflow.python.framework.ops import Graph
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.preprocessing.normalization import Normalization
from tensorflow.python.ops.gen_math_ops import Real, square
from tensorflow.python.ops.image_ops_impl import random_flip_left_right
from tensorflow.python.keras.backend import ones_like, squeeze, zeros
from tensorflow.python.keras.engine.training import concat
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, ReLU, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Reshape, Dropout, Concatenate, Lambda, Multiply, Add, Flatten, Dense, Conv2DTranspose, GaussianNoise
# from tensorflow.keras.layers import RandomFlip, RandomTranslation, RandomRotation, RandomZoom
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
from skimage.color import rgb2ycbcr, ycbcr2rgb
from matplotlib import cm, pyplot as plt
from timeit import default_timer
from tensorboard import program
from tensorboard import main as tb
from tabulate import tabulate
import pickle


# Removes error when running Tensorflow on GPU
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# Reduces Tensorflow messages other than errors or important messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# To debug @tf.functions in vscode (otherwise breakpoints dont work)
# tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# adaptive discriminator augmentation
max_translation = 0.125
max_rotation = 0.125
max_zoom = 0.25
target_accuracy = 0.85
integration_steps = 1000

# ------------------------------------------------
# =============== THE MAIN CLASS ===============
# ------------------------------------------------
class SpecSeg( object ):
    # ------------------------------------------------
    #
    # ██╗███╗   ██╗██╗████████╗
    # ██║████╗  ██║██║╚══██╔══╝
    # ██║██╔██╗ ██║██║   ██║
    # ██║██║╚██╗██║██║   ██║
    # ██║██║ ╚████║██║   ██║
    # ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝
    # ------------------------------------------------
    def __init__( self, args ):

        # only create Comet experiment if training, not testing
        # Create an experiment with your api key
        if args.mode == 'train':
            self.comet_experiment = Experiment(
                api_key                            = "doJU3H6SCSuhOYCdYC4s50olk",
                project_name                       = "specseg",
                workspace                          = "atifanwer",
                auto_param_logging                 = True,
                auto_metric_logging                = True,
                log_env_details                    = True,
                log_code                           = True,                        # code logging
                log_graph                          = True,
                log_env_gpu                        = True,
                log_env_host                       = True,
                log_env_cpu                        = True,
                auto_histogram_tensorboard_logging = True,
                auto_histogram_weight_logging      = True,
                auto_histogram_gradient_logging    = True,
                auto_histogram_activation_logging  = True,
                # auto_histogram_epoch_rate=1,
            )
            self.comet_experiment.add_tag("SpecSeg_test")

        # Model configuration.
        self.c_dim        = args.c_dim
        self.image_size   = args.image_size  # the size of the image after resizing

        # Training configuration.
        self.batch_size          = args.batch_size  # batch size for training
        self.num_epochs          = args.num_epochs
        self.num_iteration_decay = args.num_iteration_decay
        self.g_lr                = args.g_lr
        self.d_lr                = args.d_lr
        self.n_critic            = args.n_critic
        self.beta1               = args.beta1
        self.beta2               = args.beta2
        self.d_repeat_num        = args.d_repeat_num

        # Test configurations.
        # self.test_iters = args.test_iters

        # Miscellaneous.
        self.mode = args.mode

        # Directories.
        self.data_dir            = args.data_dir  # the root folder containing polarimetric sub-folders
        self.model_save_dir      = args.model_save_dir
        self.checkpoint_save_dir = args.checkpoint_save_dir
        self.result_dir          = args.result_dir
        self.log_dir             = args.log_dir
        # self.lambda_recons       = args.lambda_recons
        # self.lambda_class        = args.lambda_class

        # Step size.
        self.log_step             = args.log_step
        self.checkpoint_save_step = args.checkpoint_save_step

        # Misc parameters
        self.filter_size  = args.filter_size
        self.seed         = 25
        self.randomness   = 0.50
        self.dropout_amnt = 0.2  # ( 0.2 used in CollaGAN)

        # CollaGAN uses this for calculating the GAN loss... no idea why (yet)
        self.TARGET_LABELS = 0.90  # - Label smoothing by not using hard 1.0 value
        # To use LSGAN
        self.use_lsgan = True

        self._graph=tf.Graph()

        # using different optimizers for G and D
        # To use decayed learning rate, replace the LR with this
        decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(   initial_learning_rate = self.g_lr, \
                                                                        decay_steps = 10000, decay_rate = 00.95, \
                                                                        staircase = False )

        self.optimizer_unet = Adam( learning_rate = decayed_lr, beta_1 = self.beta1, beta_2 = self.beta2 )

        """
        Keep or delete old checkpoints
        """
        self.delete_old_checkpoints = args.delete_old_checkpoints

        # Only traing Discriminator at first. Train Generator after n epochs
        self.train_G_after = 0

        self.c_dim        = 5
        self.g_conv_dim   = 64
        self.g_repeat_num = 6

        # for plotting histograms of gradients
        self.gradmapD = {}
        self.gradmapG = {}

        self.init = RandomNormal(mean=0.0, stddev=0.02, seed=42)     # suggested by DCGAN

        # Initialize for use in training
        self.random_flip = 0.0

        # initializing the specular candidate matrix
        self.specular_candidate = tf.zeros( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )

        # Trainable weights (?)
        self.gamma = tf.Variable(initial_value=0, shape=(), trainable=True, dtype=tf.float32, name="Gamma")

        # Zhao et al (2018) Loss function alpha as defined in paper
        self.alpha = 0.84


        # # Nvidia ADA
        # # Sauce: https://keras.io/examples/generative/gan_ada/
        # self.augmenter = AdaptiveAugmenter()


    # ------------------------------------------------
    #
    #  ██████  ███████ ███    ██ ███████ ██████   █████  ████████  ██████  ██████
    # ██       ██      ████   ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
    # ██   ███ █████   ██ ██  ██ █████   ██████  ███████    ██    ██    ██ ██████
    # ██    ██ ██      ██  ██ ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
    #  ██████  ███████ ██   ████ ███████ ██   ██ ██   ██    ██     ██████  ██   ██
    #
    # Generator has two inputs and one output
    # INPUT: Concatenated multiple 5x Y-channel images (with concatenated label channels)
    # OUTPUT: Single Image (Single Y channel only - concatenate with CbCr after generation) (generated_image)
    # ------------------------------------------------
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def build_unet( self ):
        # inp_images = INPUT2 >> Concatenated 5x Y Channel images
        inp_images = Input( shape = (self.image_size, self.image_size, 3) )

        # UNET Architecture inspired from CollaGAN
        # inp                                        o/p
        # └── d1 -------------------------------- u4 ──┘
        #     └── d2 ----------------------- u3 ──┘
        #         └── d3 -------------- u2 ──┘
        #             └── d4 ----- u1 ──┘
        #                   └─ im ─┘

        # DOWNSAMPLE 5 times
        N = self.filter_size  # Should always be image_size/2 (?)
        x = inp_images
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )
        down1 = x
        # self.attn_1, pooled = self.attention_layer( spec=self.specular_candidate, filter_size=N, pool=False )
        # self.attn_1, attention_map = self.sagan_attention( self.specular_candidate , filter_size=N, pool=False )
        x = AveragePooling2D( pool_size = (2, 2), strides = None, padding = 'same' )( x )
        N *= 2
        # ----d1

        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )
        down2 = x
        # self.attn_2, pooled = self.attention_layer( spec=pooled, filter_size=N, pool=True )
        # self.attn_2, attention_map = self.sagan_attention( self.specular_candidate , poolsize=(2,2), filter_size=N, pool=True )
        x = AveragePooling2D( pool_size = (2, 2), strides = None, padding = 'same' )( x )
        N *= 2
        # ----d2

        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )
        down3 = x
        # self.attn_3, pooled = self.attention_layer( spec=pooled, filter_size=N, pool=True )
        # self.attn_3, attention_map = self.sagan_attention( self.specular_candidate , poolsize=(4,4), filter_size=N, pool=True )
        x = AveragePooling2D( pool_size = (2, 2), strides = None, padding = 'same' )( x )
        N *= 2
        # ----d3

        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )
        down4 = x
        # self.attn_4, pooled = self.attention_layer( spec=pooled, filter_size=N, pool=True )
        # self.attn_4, attention_map = self.sagan_attention( self.specular_candidate , poolsize=(8,8), filter_size=N, pool=True )
        x = AveragePooling2D( pool_size = (2, 2), strides = None, padding = "same" )( x )
        # ----d4

        # Adding 1x1 conv layers
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 1, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )
        #  ----- 1x1 layers
        # N *= 2

        # Multiply the skip connection with the attention layer generated from the Mask.
        # Note that attention layer is between [0,1] after sigmoid
        # Note that the first call does not Maxpool the mask to retain dimension consistency

        # down4 = down4 * self.attn_4
        # down3 = down3 * self.attn_3
        # down2 = down2 * self.attn_2
        # down1 = down1 * self.attn_1

        # # UPSAMPLE
        # ----u1
        # N /= 2
        x = Conv2DTranspose( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu  )( x )
        x = Concatenate()( [x, down4] )
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init )( x )
            # x = LeakyReLU()( x )
        # ----u2
        N /= 2
        x = Conv2DTranspose( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu  )( x )
        x = Concatenate()( [x, down3] )
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
        #     x = LeakyReLU()( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )
        # ----u3
        N /= 2
        x = Conv2DTranspose( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu  )( x )
        x = Concatenate()( [x, down2] )
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )
        # ----u4
        N /= 2
        x = Conv2DTranspose( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu  )( x )
        x = Concatenate()( [x, down1] )
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )

        # Output is a single Y channel image
        genOutput = Conv2D( filters = 1, kernel_size = 1, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.sigmoid)( x )
        return Model( inp_images, genOutput, name = 'SHM_Unet' )

    
    #
    #  █████  ████████ ████████ ███████ ███    ██ ████████ ██  ██████  ███    ██
    # ██   ██    ██       ██    ██      ████   ██    ██    ██ ██    ██ ████   ██
    # ███████    ██       ██    █████   ██ ██  ██    ██    ██ ██    ██ ██ ██  ██
    # ██   ██    ██       ██    ██      ██  ██ ██    ██    ██ ██    ██ ██  ██ ██
    # ██   ██    ██       ██    ███████ ██   ████    ██    ██  ██████  ██   ████
    #
    # ----------------------------------------------
    # Generating the attention layer from the specular candidate from Shen function
    # Note that the filter size is provided on call
    # NOTE: Sigmoid activation included in the layer.
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def attention_layer( self, spec, filter_size, poolsize=(2,2) ,pool=True ):
        if pool == True:
            pooled = MaxPooling2D( pool_size = poolsize, strides = None, padding = "same" )( spec )
        else:
            pooled = self.specular_candidate

        spec = Conv2D( filters = filter_size, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( pooled )
        spec = Conv2D( filters = filter_size, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( spec )
        return spec, pooled

    """
    # ------------------------------------------------
    #
    # ████████ ██████   █████  ██ ███    ██       ███████ ████████ ███████ ██████
    #    ██    ██   ██ ██   ██ ██ ████   ██       ██         ██    ██      ██   ██
    #    ██    ██████  ███████ ██ ██ ██  ██ █████ ███████    ██    █████   ██████
    #    ██    ██   ██ ██   ██ ██ ██  ██ ██            ██    ██    ██      ██
    #    ██    ██   ██ ██   ██ ██ ██   ████       ███████    ██    ███████ ██
    #
    #                   BUILD THE MODEL
    # ------------------------------------------------
    """
    # jit_compile=True
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def train_step( self, input_image, input_mask ):

        # create zeros and ones for labels
        tmp_zeros = tf.zeros( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )
        trg_ones  = tf.ones( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )

        self.target_label_0deg   = tf.Variable( [self.TARGET_LABELS,0,0,0,0],  dtype=tf.float32 )

        # using separate tapes for both D and G (sauce: pix2pix tutorial from TF site)
        with tf.GradientTape( persistent=True ) as tape:

            ds1_yuv = tf.image.rgb_to_yuv( input_image[:, :, :, :] )

            self.origIm_yuv   = ds1_yuv[:, :, :, 0, tf.newaxis]


            # calculate the specular candidate image from the Y-channel. Instead of Itot, just using I0 as input for now.
            # Note: Rescaling the Y-channel to 0-1 before calcualting specular candidate
            # self.specular_candidate = self.Shen2009_specular_candidate( self.origIm_yuv )

            # self.specular_candidate = tf.image.rgb_to_grayscale( input_mask )

            # averaging the CB and CR channels to get the estimated CbCr to use later
            # TODO: Test if we can replace with minimum?
            CbCr = ( self.origIm_yuv[:, :, :, 1:] )
            CbCr = tf.cast(CbCr, tf.float32)

            # generating random numbers
            RNG1 = tf.random.uniform( [] ) < self.randomness

            # plt.imshow(tf.squeeze( ds1[:, :, :, 0, tf.newaxis] ), cmap=cm.gray)       - Works

            # Generate only 1 Y channel image, based on random target class.
            if self.epoch > self.train_G_after:
                trainingFlag = True
            else:
                trainingFlag = False


            self.gen_mask   = self.unet ( input_image, training=trainingFlag )


            # ==================================================================
            #
            # ██       ██████  ███████ ███████ ███████ ███████
            # ██      ██    ██ ██      ██      ██      ██
            # ██      ██    ██ ███████ ███████ █████   ███████
            # ██      ██    ██      ██      ██ ██           ██
            # ███████  ██████  ███████ ███████ ███████ ███████
            # ==================================================================

            dice_loss = self.dice_loss( self.gen_mask, input_mask )

            """----------------Losses--------------------"""

            # aka G_gan_loss_cyc
            # D3_RealFake_cyc1 = tf.math.reduce_mean( tf.math.squared_difference( self.inp_rgb, self.TARGET_LABELS ) )

            # -------------------------------------------------

            # oneHot lables will be of shape (1, 5)
            # oneHot_lbl1 = tf.reshape(tf.one_hot(tf.cast(0,tf.uint8),5),[1,5])
            # Categorical crossentropy = Softmax crossentropy (for multi-class classification)
            # D3_classification_loss1 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl1, logits = label_cyc_gen0_D3   )

            # mask_loss = tf.nn.sigmoid_cross_entropy_with_logits( labels = oneHot_lbl1, logits = self.inp_rgb)

            # # The Least Squares Generative Adversarial Network, or LSGAN for short, is an extension to the GAN architecture that addresses the problem of vanishing gradients and loss saturation.
            # # Using LSGAN for D2 and D4 outputs:
            # # D_loss = 0.5 * tf.reduce_mean((D_real-1)^2) + tf.reduce_mean(D_fake^2)
            # # G_loss = 0.5 * tf.reduce_mean((D_fake -1)^2)
            # #  D2 + D1
            # self.D2_RealFake_target = tf.math.reduce_mean( tf.math.squared_difference( self.RealFake_target_D2 , self.TARGET_LABELS )) + tf.math.reduce_mean( tf.math.square( self.RealFake_gen_D1 ) )
            # # D4 + D3
            # D4_1 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_orig0_D4 , self.TARGET_LABELS ))   + tf.math.reduce_mean( tf.math.square( RealFake_cyc_gen0_D3 ) )
            # D4_2 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_orig45_D4 , self.TARGET_LABELS ))  + tf.math.reduce_mean( tf.math.square( RealFake_cyc_gen45_D3 ) )
            # D4_3 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_orig90_D4 , self.TARGET_LABELS ))  + tf.math.reduce_mean( tf.math.square( RealFake_cyc_gen90_D3 ) )
            # D4_4 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_orig135_D4 , self.TARGET_LABELS )) + tf.math.reduce_mean( tf.math.square( RealFake_cyc_gen135_D3 ) )
            # D4_5 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_origED_D4 , self.TARGET_LABELS ))  + tf.math.reduce_mean( tf.math.square( RealFake_cyc_genED_D3 ) )
            # self.D4_RealFake_cyc = D4_1 + D4_2 + D4_3 + D4_4 + D4_5 + self.D2_RealFake_target

            # D_loss = (self.D4_RealFake_cyc)/6.0 + (self.D4_classification_loss)/5.0

            # --------------------      L1 Loss       --------------------
            #  The cycle consistency loss is defined as the sum of the L1 distances between the real images from each domain and their generated counterparts.
            # Source: https://github.com/AlamiMejjati/Unsupervised-Attention-guided-Image-to-Image-Translation/blob/master/losses.py
            # L1_loss_G1    = tf.reduce_mean( tf.abs( self.gen_mask        - self.target_img ))

             # --------------------      SPECULAR       --------------------
            # Specular Loss - To foce ED generation
            # Spec_loss1  = tf.reduce_mean( tf.math.square ( (cyc_gen0_yuv   * self.specular_candidate) - (ds1_yuv * self.specular_candidate ) ) )

            # --------------------      TOTAL LOSSES       --------------------
            self.total_SpecSeg_loss      =    ( dice_loss )
                                                # ( self.total_NST_loss ) * 10.0
                                                # ( self.MS_SSIML1_loss )                                + \

        """
        # ------------------------------------------------
        #       GRADIENT-TAPE FOR LEARNING PARAMETERS
        # Calculate the gradients for generator and discriminator
        # NOTE: Gradient tape are calculated after the tf.Gradientape() function ends
        # ------------------------------------------------
        """
        # Unfreeze the weights for applying gradients - Important!
        self.unet.trainable = True
        scaled_gradient_unet = tape.gradient( [ self.total_SpecSeg_loss], self.unet.trainable_variables )
        # scaled_gradient_Generator = [(tf.clip_by_value(grad, clip_value_max=1.0, clip_value_min=-1.0)) for grad in scaled_gradient_Generator]
        # unscaled_gradient_Generator = self.optimizer_G.get_unscaled_gradients(scaled_gradient_Generator)
        self.optimizer_unet.apply_gradients( zip( scaled_gradient_unet, self.unet.trainable_variables ), experimental_aggregate_gradients=True )
        self.gradmapUnet  = scaled_gradient_unet

        return

    # ------------------------------------------------
    #
    # ███    ███  █████  ██ ███    ██
    # ████  ████ ██   ██ ██ ████   ██
    # ██ ████ ██ ███████ ██ ██ ██  ██
    # ██  ██  ██ ██   ██ ██ ██  ██ ██
    # ██      ██ ██   ██ ██ ██   ████
    # ------------------------------------------------
    # jit_compile=True
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def train( self, args ):
        # Set to train on CPU
        # os.environ["CUDA_VISIBLE_DEVICES"]="-1"

        # trainTime = time.time()
        start_time = default_timer()


        file_writer = tf.summary.create_file_writer( self.log_dir )

        tf.Graph.finalize( self._graph )

        # ------------------------------------------------
        # Initialize and load the zipped dataset
        # NOTE: Images returned will be resized, Normalized (TBD) and randomly flipped (TBD)
        self.length_dataset, Dataset = self.datasetLoad()

        # NOTE: Batch size is 1 for zipped dataset, so that 1 image from each polar angle is picked
        # batched_dataset = Dataset.batch( 4 )

        # ------------------------------------------------

        self.unet = self.build_unet( )
        # self.D = self.build_discriminator( )
        # Print Model summary to console and file
        self.unet.summary()
        # self.D.summary()
        with open('Generator_summary.txt', 'w') as f:
            self.unet.summary(print_fn=lambda x: f.write(x + '\n'))

        # plot the model
        # plot_model( self.unet, to_file = 'unet_plot.png', show_shapes = True, show_layer_names = True )

        # self.G.save('G2.h5')

        # ------------------------------------------------
        # Initiialize the checkpoint manager
        checkpoint_dir    = self.checkpoint_save_dir
        ckpt = tf.train.Checkpoint( generator     = self.unet,
                                    optimizer_D   = self.optimizer_unet)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

        # if old checkpoints exist and flag is true; del them
        if self.delete_old_checkpoints == True and ckpt_manager.latest_checkpoint:
            os.system("echo CLEANUP: Removing previous checkpoints")
            os.system("rm /home/atif/Documents/checkpoints/*")
        elif self.delete_old_checkpoints == False:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

        # ------------------------------------------------
        # Make the iterator from the zipped datasets
        iterator = iter(Dataset)

        batches_per_epoch = int( self.length_dataset/self.batch_size )
        # Epoch -> passes over the dataset

        self.batch_step = 0

        # ------------------------------------------------
        # tf.summary.trace_on(graph=True, profiler=True)
        # ------------------------------------------------

        plt.close("all")
        gc.collect()

        # self.comet_experiment.set_model_graph(self.D)
        # self.comet_experiment.set_model_graph(self.G)

        for epoch in range(self.num_epochs):

            self.epoch = epoch
            print(f"\nStart of Training Epoch {self.epoch}")
            # since batch will be 1 (to get all polar images in sequence), the number of batches
            # will have to be equal to the length of the dataset. Also note that the randomness and shuffle
            # are not in the dataset while loading, but introduced by randomizing input channels
            # and randomizing lables while training (tf.cond ....)


            for batch in range( batches_per_epoch-1 ):


                self.batch_step +=1


                # # Randomly flip all images in the batch
                # self.random_flip = tf.random.uniform( [], dtype=tf.float16 ) >= 0.5

                # Randomly generate target labels for more robustness instead of a hard value of 1
                # self.TARGET_LABELS = tf.random.uniform( [],  minval=0.8, maxval=1.2, dtype=tf.float32 )


                # Get the next set of images
                element = iterator.get_next()

                input_image = element[0]
                input_mask  = element[1]


                self.train_step( input_image, input_mask )

                # for tensorboard (check if can be moved)
                with file_writer.as_default():

                    # # # Image grid
                    # figure1 = image_grid( self.cyc_gen0_rgb, self.cyc_gen45_rgb, self.cyc_gen90_rgb, self.cyc_gen135_rgb, self.cyc_genED_rgb )
                    # figure2 = image_grid( orig0, orig45, orig90, orig135, origED )
                    # figure3 = image_grid( self.attention_map1, self.attention_map2, self.attention_map3, self.attention_map4, self.attention_map4 )

                    # plt.close("all")

                    # # ---------- COMET.ML -------------
                    # Generating a confusion matrix for target angle and predicteabels )

                    # Log some metrics but only 1 time and not every loop
                    # if self.batch_step == 1:
                        # logging other parameters for record peurposes
                        # self.comet_experiment.log_other( value = self.train_G_after, key="Train G after n loops")
                        # self.comet_experiment.log_other( value = self.dropout_amnt, key="Dropout amount")


                    # # Monitoring Losses
                    # log everything every n steps other than confusion matrices. Reduces time per epoch
                    if self.batch_step % 25 == 0:
                        self.comet_experiment.log_metric ( "Total Unet Loss", tf.squeeze(self.total_SpecSeg_loss ), step=self.batch_step, epoch = self.epoch )
                        # self.comet_experiment.log_metric ( "NST Loss", tf.squeeze(self.total_NST_loss), step=self.batch_step, epoch = self.epoch )

                        # Plotting output of G1
                        self.comet_experiment.log_image( tf.squeeze( (input_image) ), name="Input RGB",        step=self.batch_step)
                        self.comet_experiment.log_image( tf.squeeze( (input_mask ) ), name="Input Mask",       step=self.batch_step)
                        self.comet_experiment.log_image( tf.squeeze( (self.gen_mask) ), name="Generated Mask", step=self.batch_step)

                        # plt.close("all")
                        # gc.collect()

                    # Onlt log histogram every n steps, to avoid comet upload rate issue
                    # alos doesnt slow down training
                    if self.batch_step % 100 == 0:
                        for index2, grad2 in enumerate(self.gradmapUnet):
                            self.comet_experiment.log_histogram_3d( self.gradmapUnet[index2], name="Unet Grads", step=self.batch_step, epoch=self.epoch)

                    # CLEAR UP MEMORY
                    file_writer.flush()
                    plt.close("all")
                    gc.collect()

                    # print the progress (taken from stargan-github)
                    printProgressBar( (batch+1) % 1000, batches_per_epoch-1, decimals=2)

             #  Print losses every x epochs
            if (self.epoch + 1) % self.log_step == 0:
                finish = default_timer()
                print( "\n")
                # print( f"\tIteration: [{self.epoch + 1}/{self.log_step}]" )
                print ('Time taken for epoch {} is {} min\n'.format(self.epoch + 1, (finish-start_time)/60))




                # In eager mode, grads does not have name, so we get names from model.trainable_weights
                # for weights, grads in zip(self.D.trainable_weights, self.gradmapD):
                #     tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads, step=self.batch_step)

                # gc.collect()

                # Stop the trace and export the collected information
                # tf.summary.trace_export(name="Train", step=self.batch_step, profiler_outdir=self.log_dir)

            # Save the weights at every x steps
            if (self.epoch + 1) % self.checkpoint_save_step == 0:
                # save checkpoint -
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(self.epoch+1, ckpt_save_path))

        print( f"Time for training was {(time.time() - start_time) / 60.0} minutes" )

        # Save last checkpoint before quitting
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(self.epoch+1, ckpt_save_path))
        file_writer.flush()
        gc.collect()
        # close tensorboard writer
        file_writer.close()
        return


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
    def datasetLoad( self ):
        rootfolder = self.data_dir
        # FOR SHMGAN Dataset
        path1 = os.path.join( rootfolder, 'HighlightImages' )
        path2 = os.path.join( rootfolder, 'HighlightMasks' )

        data_dir1 = pathlib.Path( path1 )
        data_dir2 = pathlib.Path( path2 )

        # NOTE:  => The generated datasets do not have any lables

        rgb_images = tf.keras.preprocessing.image_dataset_from_directory(
                str( data_dir1 ),
                  labels           = None,
                # label_mode       = 'categorical',
                  color_mode       = 'rgb',
                  validation_split = None,
                  shuffle          = False,
                  seed             = 1337,
                  image_size       = (self.image_size, self.image_size),
                  batch_size       = 1
                ) \
                .cache() \
                .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
                .prefetch(25)
                # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                # .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
        # Manually update the labels (?)
        rgb_images.class_names = 'RGB'

        masks = tf.keras.preprocessing.image_dataset_from_directory(
                str( data_dir2 ),
                  labels           = None,
                # label_mode       = 'categorical',
                  color_mode       = 'rgb',
                  validation_split = None,
                  shuffle          = False,
                  seed             = 1337,
                  image_size       = (self.image_size, self.image_size),
                  batch_size       = 1
                ) \
                .cache() \
                .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
                .prefetch(25)
                # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                # .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
        masks.class_names = 'MASK'

        # ZIP the datasets into one dataset
        loadedDataset = tf.data.Dataset.zip ( ( rgb_images, masks ) )
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
        self.loadedDataset = loadedDataset.cache().repeat( self.num_epochs ).prefetch( buffer_size =25)
        # -------------------------------------------------------

        # return the number of files loaded , to calculate iterations per batch
        self.length_dataset = len(np.concatenate([i for i in rgb_images], axis=0))
        # returns the zipped dataset for use with iterator
        return self.length_dataset, self.loadedDataset

    """
    # ------------------------------------------------
    #
    # ████████ ███████ ███████ ████████
    #    ██    ██      ██         ██
    #    ██    █████   ███████    ██
    #    ██    ██           ██    ██
    #    ██    ███████ ███████    ██
    #
    # TEST FUNCTION FOR SHMGAN
    # The test function has the following features:
    # - Load RGB image from test folder as I0 (or Itot)
    # - Set all other layers to zero
    # - Set target image label as ED
    # - Average CbCr is replaced with CbCr of the image
    # - Generate images. Both G1 and G_cyclic
    # - No need for losses
    # ------------------------------------------------
    """
    def test( self, args ):

        # self.comet_experiment = Experiment(
        #         api_key                            = "doJU3H6SCSuhOYCdYC4s50olk",
        #         project_name                       = "shm",
        #         workspace                          = "atifanwer",
        #         auto_param_logging                 = True,
        #         auto_metric_logging                = True,
        #         log_env_details                    = True,
        #         log_code                           = True,                        # code logging
        #         log_graph                          = True,
        #         log_env_gpu                        = True,
        #         log_env_host                       = True,
        #         log_env_cpu                        = True,
        #         auto_histogram_tensorboard_logging = True,
        #         auto_histogram_weight_logging      = True,
        #         auto_histogram_gradient_logging    = True,
        #         auto_histogram_activation_logging  = True,
        #         # auto_histogram_epoch_rate=1,
        #     )
        # self.comet_experiment.add_tag("SPECSEG TEST RUN")

        # Do not flip the image
        self.random_flip = 0.0
        # do not randomize target label values
        self.TARGET_LABELS = 1.0

        # Disable deleting by mistake
        self.delete_old_checkpoints = False

        # Step1: Load the test images
        rootfolder = args.test_dir
        testpath = pathlib.Path( rootfolder )
        # NOTE: While loading the images, only difference is that the images are not flipped. Otherwise it is the same function as
        # the dataset loading images
        test_images = tf.keras.preprocessing.image_dataset_from_directory(
                str( testpath ),
                  labels           = None,
                # label_mode       = 'categorical',
                  color_mode       = 'rgb',
                  validation_split = None,
                  shuffle          = False,
                  seed             = 1337,
                  image_size       = (self.image_size, self.image_size),
                  batch_size       = 1
                ) \
                .cache() \
                .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE ) \
                .map(lambda x: tf.image.per_image_standardization( x ) ) \
                .prefetch(25)
                # .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE ) \
        test_images.class_names = 'TEST'

        # Only load diffuse images if the flag is True
        if args.calc_metrics == True:
            diffusefolder = args.diffuse_dir
            diffusepath = pathlib.Path( diffusefolder )
            # NOTE: While loading the images, only difference is that the images are not flipped. Otherwise it is the same function as
            # the dataset loading images
            diffuse_images = tf.keras.preprocessing.image_dataset_from_directory(
                    str( diffusepath ),
                    labels           = None,
                    # label_mode       = 'categorical',
                    color_mode       = 'rgb',
                    validation_split = None,
                    shuffle          = False,
                    seed             = 1337,
                    image_size       = (self.image_size, self.image_size),
                    batch_size       = 1
                    ) \
                    .cache() \
                    .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE ) \
                    .map(lambda x: tf.image.per_image_standardization( x ) ) \
                    .prefetch(25)
                    # .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                    # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE ) \
            test_images.class_names = 'TEST'

        # return the number of files loaded
        self.number_of_test_images = len(np.concatenate([i for i in test_images], axis=0))

        # ZIP the datasets into one dataset
        if args.calc_metrics == True:
            loadedDataset = tf.data.Dataset.zip ( ( test_images, diffuse_images ) )
        else:
            loadedDataset = tf.data.Dataset.zip ( test_images )

        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        loadedDataset = loadedDataset.with_options(options)
        loadedDataset = loadedDataset.cache().prefetch( buffer_size =25)

        # Load the G and D
        self.G = self.build_unet( )
        # self.D = self.build_discriminator( )
        # Print Model summary to console and file
        self.G.summary()
        self.D.summary()
        with open('Generator_summary.txt', 'w') as f:
            self.G.summary(print_fn=lambda x: f.write(x + '\n'))
        with open('Discriminator_summary.txt', 'w') as f:
            self.D.summary(print_fn=lambda x: f.write(x + '\n'))


        # STEP2: Load checkpoints
        checkpoint_dir    = self.checkpoint_save_dir
        ckpt = tf.train.Checkpoint( generator     = self.G,
                                    discriminator = self.D,
                                    optimizer_D   = self.optimizer_D,
                                    optimizer_G   = self.optimizer_G )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print ('Latest checkpoint restored!!')

        plt.close("all")
        gc.collect()

        # STEP3: Iterate over the loaded test images
        test_iterator = iter(loadedDataset)

        # create zeros and ones for labels
        tmp_zeros = tf.zeros( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )
        trg_ones  = tf.ones( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )

        # initialize lists for printing data
        MAE   = []
        MSE   = []
        SSIM  = []
        PSNR  = []
        index = []
        table = []
        processing_time_taken = []

        print('\n\n\n->> "I\'m sorry, Dave. You will have to wait a little while I process... Regards, HAL 9000 ◍ <<- \n\n\n')

        # for all images in the test folder
        for i in range(self.number_of_test_images):

            self.start_time = time.time()

            # Randomly generate target labels for more robustness instead of a hard value of 1
            self.TARGET_LABELS = tf.random.uniform( [],  minval=0.8, maxval=1.2, dtype=tf.float32 )

            # Get the image
            element = test_iterator.get_next()
            if args.calc_metrics == True:
                self.rgb_testImage   = element[0]
                self.gt  = element[1]
            else:
                self.rgb_testImage   = element

            # Setting target labels for Cyclic generation
            self.target_label_ED     = tf.Variable( [0,0,0,0,self.TARGET_LABELS],  dtype=tf.float32 )

            # setting both G and D as non-trainable
            self.G.trainable = False
            self.D.trainable = False

            # setting ED as input image and other channels as zero
            RGBInput = tf.image.rgb_to_yuv( self.rgb_testImage[:, :, :, :] )

            # Generating the specular mask from the input RGB image
            self.specular_candidate = self.Shen2009_specular_candidate( RGBInput[:, :, :, 0, tf.newaxis] )
            # plot_single_image ( self.specular_candidate, title="Specular Candidate" )

            # setting the CbCr same as the input image
            averageCbCr = RGBInput[:, :, :, 1:]

            # Y channel input are set to zero and the input is 0 degree
            ych_inp1 = RGBInput[:, :, :, 0, tf.newaxis]
            ych_inp2 = tmp_zeros
            ych_inp3 = tmp_zeros
            ych_inp4 = tmp_zeros
            ych_inp5 = tmp_zeros
            # generate the inputs
            rand_input_Ych = tf.concat( [ych_inp1, ych_inp2, ych_inp3, ych_inp4, ych_inp5], axis = 3 )

            self.gen_input          = tf.concat( [rand_input_Ych, tmp_zeros, tmp_zeros, tmp_zeros, tmp_zeros, trg_ones], axis = 3 )
            self.target_img         = self.rgb_testImage
            self.Target_angle_label = self.target_label_ED

            # test plot the input
            # debug_plot( self.gen_input )

            """--------------------G(1)-------------------"""
            self.inp_rgb   = self.G ( self.gen_input, training=False )
            gen_YCbCr   = tf.concat( [self.inp_rgb, averageCbCr], axis = 3 )
            self.gen_mask   = tf.image.yuv_to_rgb( gen_YCbCr )

            orig_Ych = self.gen_mask[:, :, :, 0, tf.newaxis]

            # plot_single_image ( self.gen_Y )
            # plot_single_image ( self.gen_rgb, title="Generated RGB" )

            processing_time_taken.append( (time.time() - self.start_time) )

            # self.test_plot()

            #  ---------------------- COMET Logging ------------------------
            # Plotting output of G1
            self.comet_experiment.log_image( tf.squeeze( (self.inp_rgb) ), name="G1 Y-ch", step=i)
            self.comet_experiment.log_image( tf.squeeze((self.gen_mask)), name="G1 RGB", step=i)
            if args.calc_metrics == True:
                self.comet_experiment.log_image( tf.squeeze(self.gt), name="2. Target Diffuse ", step=i)
            #  ---------------------- PyPlt printing ------------------------

            # image_grid( self.cyc_gen0_rgb, self.cyc_gen45_rgb, self.cyc_gen90_rgb, self.cyc_gen135_rgb, self.cyc_genED_rgb )
            # plot the generated images :fingerscrossed:
            # plot_single_image ( self.cyc_gen0_rgb )
            # plot_single_image ( self.cyc_gen45_rgb )
            # plot_single_image ( self.cyc_gen90_rgb )
            # plot_single_image ( self.cyc_gen135_rgb )
            # plot_single_image ( self.cyc_genED_rgb )
            # plt.close("all")
            # gc.collect()

            # ----------------- calculating Metrics -------------------
            # calculate only if the flag is true
            if args.calc_metrics == True:
                index.append(i+1)

                # FID_score   = self.calculate_FID( self.cyc_genED_rgb , self.target_img )
                SSIM.append( (tf.image.ssim ( rescale_01( self.gen_mask ), rescale_01( self.gt ), 5 )).numpy() )
                PSNR.append( (tf.image.psnr ( self.gen_mask , self.gt, max_val=255 )).numpy() )

                # Calculate L1 loss to original image?
                # Or use builtin functions to evaluate the Generator?
                L2_loss = tf.keras.losses.MeanSquaredError()
                MSE.append( L2_loss(self.gen_mask, self.gt ).numpy() )

                # print ( 'Processing Image# {}: {:.3f} secs, MSE:{:.4f}, SSIM:{:.4f}, PSNR:{:.4f} \n' .format( i, processing_time_taken[i], MSE[i], SSIM[i], PSNR[i]) )

                # populate table
                column = [index[i], processing_time_taken[i], MSE[i], SSIM[i], PSNR[i]]
                table.append(column)

                # # print table inline
                # print( tabulate( table, tablefmt="plain" ))

                self.comet_experiment.log_metric ( "Processing Time", processing_time_taken[i], step=i  )
                self.comet_experiment.log_metric ( "MSE", MSE[i], step=i  )
                self.comet_experiment.log_metric ( "SSIM", SSIM[i], step=i  )
                self.comet_experiment.log_metric ( "PSNR", PSNR[i], step=i  )

        # Print metrics only if flag is true
        if args.calc_metrics == True:
            print('\n\n --- PRINTING ALL CALCUATED METRICS --- ')
            print(tabulate(table, headers=['Image#', 'Time', 'MSE', 'SSIM', 'PSNR']))

            # Calculating mean values
            mean_mse  = sum(MSE) / len(MSE)
            mean_ssim = sum(SSIM) / len(SSIM)
            mean_psnr = sum(PSNR) / len(PSNR)
            print('\n\n --- PRINTING MEAN METRICS --- ')
            mean_metrics = [mean_mse, mean_ssim, mean_psnr]
            print(tabulate([mean_metrics], headers=['Mean MSE', 'Mean SSIM', 'Mean PSNR']))
            print('\n\n' )

            # saving all the calculated metrics as txt
            with open("SSIM.txt", 'wb+') as file1:
                pickle.dump(SSIM, file1)

            with open("MSE.txt", 'wb+') as file2:
                pickle.dump(MSE, file2)

            with open("PSNR.txt", 'wb+') as file3:
                pickle.dump(PSNR, file3)

            # logging means to Comet also before closing experiment
            # self.comet_experiment.log_other( value = MSE, key="All MSE")
            # self.comet_experiment.log_other( value = SSIM, key="All SSIM")
            # self.comet_experiment.log_other( value = PSNR, key="All PSNR")
            self.comet_experiment.log_other( value = mean_mse,  key="Mean MSE")
            self.comet_experiment.log_other( value = mean_ssim, key="Mean SSIM")
            self.comet_experiment.log_other( value = mean_psnr, key="Mean PSNR")

        self.comet_experiment.end()

        print('\n\n\n->> "Thank you for a very enjoyable game - HAL 9000 ◍ <<- \n\n\n')



        return

    # ------------------------------------------
    # PLOTTING TEST IMAGES POST TRAINING
    def test_plot( self ):
        figure = plt.figure( figsize=(10,15) )
        figure.add_subplot( 2, 1, 1, title="Orig")
        plt.imshow ( tf.squeeze( rescale_01( self.rgb_testImage ) ).numpy().astype("float32") )
        figure.add_subplot( 2, 1, 2, title="Generated G1")
        plt.imshow ( tf.squeeze( rescale_01( self.gen_mask ) ).numpy().astype("float32") )



    # ------------------------------------------
    #
    # ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
    # ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
    # █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
    # ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
    # ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████
    #
    # ------------------------------------------

    # ------------------------------------------
    # Implementation of customized Shen et al. (2009) paper. The specular candidate is calculated on the Y channel as input.
    # This specular candidate acts as input for the Attention Aware SHM.
    # Implementation reference: https://github.com/vitorsr/SIHR/tree/master/Shen2009
    @tf.function( experimental_follow_type_hints=True, jit_compile=True )
    def Shen2009_specular_candidate( self, y_channel ):
        # As suggested in the paper
        nu = 1.5

        # add gaussian blur to smooth the image before thresholding
        # NOTE: Maybe remove the blur since it causes blurring in the generated output due to multiplication
        # y_channel = tfa.image.gaussian_filter2d(y_channel, sigma=0.5)
        # NOTE: Rescaling to 0-1 casues grayscale mask... so dont.
        images = tf.reshape( y_channel , [self.image_size*self.image_size, 1])

        # Calculate specular-free image
        # T_v = mean(I) + nu * std(I)
        T_v = tf.math.add(tf.math.reduce_mean(images) , tf.math.multiply(nu, tf.math.reduce_std(images)))

        # Calculate specular component
        # beta_s = (I_min - T_v) .* (I_min > T_v) + 0;
        # beta_s = tf.math.multiply( tf.math.subtract( images, T_v ) , (images > T_v))
        boolean_Tensor = (images >= T_v)
        beta_s = tf.where( condition=boolean_Tensor, y=images, x=tf.math.subtract( images, T_v ) )
        specular_candidate = tf.expand_dims(tf.reshape(beta_s, [self.image_size, self.image_size,1]), axis=0);

        # subtract the diffuse image returned from the original y-channel and pass it through a sigmoid for [0,1] range
        # specular_candidate = tf.nn.relu( tf.math.sign( tf.math.subtract((y_channel), specular_candidate) ) )
        specular_candidate = rescale_01( ( tf.math.subtract((y_channel), specular_candidate) ) )

        # plt.imshow(tf.squeeze(specular_candidate).numpy().astype("float32"), cmap=cm.gray)
        # plt.axis("off")
        # plt.show()

        return specular_candidate

    # ------------------------------------------
    #              INCEPTION SCORE
    # ------------------------------------------
    # Sauce: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
    def calculate_inception_score( self, images, n_split=10, eps=1E-16 ):
        n_part = floor(images.shape[0] / n_split)
        for i in range(n_split):
            # retrieve p(y|x)
            ix_start, ix_end = i * n_part, i * n_part + n_part
            p_yx = yhat[ix_start:ix_end]
            # calculate p(y)
            p_y = expand_dims(p_yx.mean(axis=0), 0)
            # calculate KL divergence using log probabilities
            kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
            # sum over classes
            sum_kl_d = kl_d.sum(axis=1)
            # average over images
            avg_kl_d = mean(sum_kl_d)
            # undo the log
            is_score = exp(avg_kl_d)
            # store
            scores.append(is_score)
        # average across images
        is_avg, is_std = mean(scores), std(scores)
        return is_avg, is_std

        # pretend to load images
        images = ones((50, 299, 299, 3))
        print('loaded', images.shape)
        # calculate inception score
        is_avg, is_std = calculate_inception_score(images)
        print('score', is_avg, is_std)

    def dice_loss( self, y_true, y_pred ):
        # y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return (1 - numerator / denominator)


    # ------------------------------------------------
    # REFERENCES:
    # ------------------------------------------------
    # https://www.jeremyjordan.me/semantic-segmentation/
    # https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html