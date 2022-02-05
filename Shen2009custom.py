
import numpy as np
import os
import time
import datetime
import pathlib
import gc
from matplotlib import cm
import tensorflow_addons as tfa
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.layers import Reshape
import matplotlib.pyplot as plt

# def Shen2009custom(I, n_row, n_col):
# %Shen2009 I_d = Shen2009(I)
# %  This method works by finding the largest highlight area, dilating it to
# %  find the surrounding region, and then finding a coefficient that scales
# %  how much the pseudo specular component will be subtracted from the
# %  original image to find I_d.
# %
# %  The nomenclature is in accordance with the corresponding paper with
# %  exception of using I* instead of V* to denote an image.
# %
# %  See also SIHR, Shen2008, Shen2013.

def main():
    rootfolder = '/home/atif/Documents/Datasets/PSD_Dataset/PSDpolarForSHMGAN'
    path1 = os.path.join( rootfolder, 'I90' )
    data_dir1 = pathlib.Path( path1 )

    # NOTE:  => The generated datasets do not have any lables

    train_ds_0 = tf.keras.preprocessing.image_dataset_from_directory(
            str( data_dir1 ),
                labels           = None,
            # label_mode       = 'categorical',
                color_mode       = 'rgb',
                validation_split = None,
                shuffle          = False,
                seed             = 1337,
                image_size       = (128, 128),
                batch_size       = 1
            ) \
            .cache() \
            .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
            .map(lambda x: tf.image.per_image_standardization( x ) ) \
            .prefetch(300) \
            .repeat()
    # Manually update the labels (?)
    train_ds_0.class_names = 'I0'

    i = 0
    for images in train_ds_0.take(750).cache().repeat():
        q = tf.image.rgb_to_yuv(images)
        y_ch = rescale_01(q[:, :, :, 0, tf.newaxis])
        # y_ch = tfa.image.gaussian_filter2d(y_ch)
        nu = 1.5
        images = tf.reshape(y_ch, [128*128, 1])

        # Calculate specular-free image
        # T_v = mean(I) + nu * std(I)
        T_v = tf.math.add(tf.math.reduce_mean(images) , tf.math.multiply(nu, tf.math.reduce_std(images)))

        # Calculate specular component
        # beta_s = (I_min - T_v) .* (I_min > T_v) + 0;
        # beta_s = tf.math.multiply( tf.math.subtract( images, T_v ) , (images > T_v))
        boolean_Tensor = (images >= T_v)
        beta_s = tf.where( condition=boolean_Tensor, y=images, x=tf.math.subtract( images, T_v ) )
        specular_candidate = tf.expand_dims(tf.reshape(beta_s, [128, 128,1]), axis=0);

        # subtract the diffuse image returned from the original y-channel and pass it through a sigmoid for [0,1] range
        # specular_candidate = tf.nn.relu( tf.math.sign( tf.math.subtract((y_channel), specular_candidate) ) )
        specular_candidate = rescale_01( ( tf.math.subtract((y_ch), specular_candidate) ) )


        ax = plt.subplot(3, 3, i + 1)
        i += 1
        plt.imshow(tf.squeeze(specular_candidate).numpy().astype("float32"), cmap=cm.gray)
        plt.axis("off")
        if i == 9:
            plt.show()
            i = 0
            plt.close("all")
            gc.collect()




def rescale_01( input_tensor ):
    # from [-1,1] to [0,1]
    # de_normalized = (input_tensor + 1.0) * 127.5 / 255.0

    # from [min, max] to [0,1]
    # normalize_value = (value − min_value) / (max_value − min_value)
    rescaled_01 = tf.math.divide_no_nan( \
                    tf.math.subtract(input_tensor,tf.math.reduce_min(input_tensor) ) , \
                    tf.math.subtract( tf.math.reduce_max(input_tensor), tf.math.reduce_min(input_tensor)))
    # de_normalized = tf.clip_by_value( de_normalized, clip_value_min=0, clip_value_max=1)
    return rescaled_01


# ------------------------------------------------
if __name__ == "__main__":
    main()