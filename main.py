"""
# -----------------------------------------------------------
# SHMGAN -  Removal of Specular Highlights by a GAN Network
#

# Uses Packages:
#     Python 3.8
#     CUDA 11.3
#     cuDnn 8.0
#     Tensorflow 2.5 + Keras 2.4

# (C) 2021 Atif Anwer, INSA Rouen, France
# Email: atif.anwer@insa-rouen.fr
# -----------------------------------------------------------
"""
import os
import argparse

# import comet_ml at the top of your file
from comet_ml import Experiment

import tensorflow as tf
# import pydot, graphviz
# import io
import matplotlib.pyplot as plt
import numpy as np
from SpecSeg import SpecSeg
# from ShmGAN_2input import ShmGAN_2input
from datetime import datetime
from tensorflow import keras
from packaging import version
from utils import check_gpu
# from tensorflow._api.v2.compat.v1 import ConfigProto
# from tensorflow._api.v2.compat.v1 import InteractiveSession
# from packaging import version

from unet_sreeni import unet_sreeni

def parse_args():
    # ---------------------------
    # TODO: FIX ARGUMENTS; REMOVE UNNECESSARY ONES AND ADD RELEVANT ONES ONLY
    # ---------------------------
    desc = "Keras implementation of GAN for specular highlight mitigation"
    parser = argparse.ArgumentParser( description = desc )

    # Flags
    parser.add_argument( '--est_diffuse', type = bool, default = True, help = '(TRUE) Estimate diffuse image from images or (FALSE) load from hdf5 file' )
    parser.add_argument( '--flip', type = bool, default = True, help = '(TRUE) Flip images randomly while loading dataset' )
    parser.add_argument( '--mode', type = str, default = 'train', choices = ['train', 'test', 'custom'] )
    parser.add_argument( '--calc_metrics', type = bool, default = False, help = '(False) Calculate metrics (PSNR, MSE, SSIM etc)' )
    parser.add_argument( '--delete_old_checkpoints', type = bool, default = True, help = '(True) Delete old checkpoints)' )
    # parser.add_argument( '--colourspace', type = str, default = 'YCbCr', choices = ['RGB', 'YCbCr'], help='To train on Y channel only or use full RGB images' )

    parser.add_argument( '--image_size', type = int, default = 128, help = 'image resize resolution' )
    parser.add_argument( '--batch_size', type = int, default = 1, help = 'mini-batch size' )
    parser.add_argument( '--num_epochs', type = int, default = 500, help = 'Number of epochs' )
    parser.add_argument( '--n_critic', type = int, default = 5, help = 'number of D updates per each G update' )
    parser.add_argument( '--log_step', type = int, default = 1, help = 'Log every x step' )
    parser.add_argument( '--checkpoint_save_step', type = int, default = 10 )

    # Model parameters
    parser.add_argument( '--filter_size', type = int, default = 64, help = 'Initial Filter size for convolution' )
    parser.add_argument( '--c_dim', type = int, default = 5, help = 'dimension of polarimetric domain images )' )
    parser.add_argument( '--g_lr', type = float, default = 0.000001, help = 'learning rate for G' )
    parser.add_argument( '--d_lr', type = float, default = 0.000001, help = 'learning rate for D' )
    parser.add_argument( '--beta1', type = float, default = 0.5, help = 'beta1 for Adam optimizer' )
    parser.add_argument( '--beta2', type = float, default = 0.99, help = 'beta2 for Adam optimizer' )
    parser.add_argument( '--num_iteration_decay', type = int, default = 100000, help = 'number of iterations for decaying lr' )
    parser.add_argument( '--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')

    # Directories.
    # parser.add_argument( '--data_dir', default = '/home/atif/Documents/Datasets/SHMGAN_dataset/PolarImages', help = 'Path to polarimetric images' )
    parser.add_argument( '--data_dir', default = '/home/atif/Documents/Datasets/WHU-specular-dataset/train', help = 'Path to Spercular images' )
    parser.add_argument( '--test_dir', default = '/home/atif/Documents/Datasets/WHU-specular-dataset/test', help = 'Path to Specular masks' )
    parser.add_argument( '--val_dir', default = '/home/atif/Documents/Datasets/WHU-specular-dataset/test', help = 'Path to Specular Validation images' )
    # parser.add_argument( '--test_dir', default = '/home/atif/Documents/Datasets/PSD_Dataset/PSD_val/PSD_val_specular', help = 'Path to polarimetric images' )
    # parser.add_argument( '--test_dir', default = '/home/atif/Documents/Datasets/PSD_Dataset/PSD_Test/PSD_Test_specular', help = 'Path to polarimetric images' )
    # parser.add_argument( '--diffuse_dir', default = '/home/atif/Documents/Datasets/PSD_Dataset/PSD_Test/PSD_Test_diffuse', help = 'Path to diffuse images' )
    parser.add_argument( '--model_save_dir', type = str, default = './models' )
    parser.add_argument( '--checkpoint_save_dir', type = str, default = '/home/atif/Documents/checkpoints' )
    parser.add_argument( '--result_dir', type = str, default = './results' )
    parser.add_argument( '--log_dir', type = str, default = './logs/train' )

    # Step size.
    parser.add_argument( '--num_iteration', type = int, default = 20000, help = 'number of total iterations for training D' )

    # cleanup the repeated ones ...
    # parser.add_argument('--lambda_l1_cyc', type=float, default=1., help='lambda_L1_cyc, StarGAN cyc loss rec')
    # parser.add_argument('--lambda_l2_cyc', type=float, default=0., help='lambda_L2_cyc, StarGAN cyc loss rec')
    # parser.add_argument('--lambda_ssim_cyc', type=float, default=10., help='lambda_ssim')
    # parser.add_argument('--lambda_l2', type=float, default=0., help='lambda_L2')
    # parser.add_argument('--lambda_l1', type=float, default=10., help='lambda_L1')
    # parser.add_argument('--lambda_ssim', type=float, default=0., help='lambda_ssim')
    # parser.add_argument('--lambda_GAN', type=float, default=1., help='lambda GAN')
    # parser.add_argument('--lambda_G_clsf', type=float, default=1., help='generator classification loss. fake to be well classified')
    # parser.add_argument('--lambda_D_clsf', type=float, default=1., help='discriminator classification loss. fake to be well classified')
    # parser.add_argument('--lambda_cyc', type=float, default=1, help='lambda_cyc')


    # print( '\n [ => ] Passing all input arguments...' )
    return parser.parse_args()

def main():
    # Parse ags
    args = parse_args()
    if len( vars (args) ) < 1:
        # check minimum arguments provided
        print(":facepalm: => Usage : main.py -data_dir TBD etc etc ")
        exit(1)

    # Set to train on GPU
    check_gpu()

    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "This notebook requires TensorFlow 2.0 or above."


    q = vars( args )
    print( '------------ Options -------------' )
    for k, v in sorted( q.items() ):
        print( '%s: %s' % (str( k ), str( v )) )
    print( '-------------- End ----------------' )


    # load the data and calc est diffuse
    # load_dataset( args )

    # Delete Previous Tensflow Logs
    os.system("echo -------------------------------")
    os.system("echo CLEANUP: Removing previous logs")
    os.system("rm ./logs/train/*")
    # os.system("rm ./logs/*")
    os.system("echo -------------------------------")

    # setup model
    # Class includes loading dataset
    specseg = SpecSeg( args )
    # shmgan = ShmGAN_2input( args )

    # # build graph
    # shmgan.build_model( )

    # train or test, as required
    if args.mode == 'train':
        # specseg.train( args )

        unet_sreeni( args )

        print( " [*] Training finished!" )
    elif args.mode == 'test':
        specseg.test( args )



# ------------------------------------------------
if __name__ == "__main__":
    # Reduces Tensorflow messages other than errors or important messages
    # '0' #default value, output all information
    # '1' #Block notification information (INFO)
    # '2' #Shield notification information and warning information (INFO\WARNING)
    # '3' #Block notification messages, warning messages and error messages (INFO\WARNING\FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()