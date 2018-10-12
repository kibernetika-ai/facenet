# MIT License
#
# Copyright (c) 2017 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Modify attributes of images using attribute vectors calculated using
'calculate_attribute_vectors.py'. Images are generated from latent variables of
the CelebA dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import math
import os
import sys
import time

import facenet
import h5py
from PIL import Image
import numpy as np
from scipy import misc
import tensorflow as tf


def main(args):
    img_mean = np.array([134.10714722, 102.52040863, 87.15436554])
    img_stddev = np.sqrt(
        np.array([3941.30175781, 2856.94287109, 2519.35791016])
    )

    vae_checkpoint = os.path.expanduser(args.vae_checkpoint)
    vae_def = importlib.import_module(args.vae_def)
    vae = vae_def.Vae(args.latent_var_size)
    gen_image_size = vae.get_image_size()

    with tf.Graph().as_default():
        filename = args.image
        # Read image and convert to 3 channels
        image = Image.open(filename)
        image = image.convert('RGB')
        image = image.resize((gen_image_size, gen_image_size), Image.ANTIALIAS)
        image = np.array(image).reshape(1, gen_image_size, gen_image_size, 3)

        images = tf.placeholder(
            tf.float32,
            shape=(None, gen_image_size, gen_image_size, 3),
            name='input'
        )

        # Normalize
        images_norm = (images - img_mean) / img_stddev

        # Resize to appropriate size for the encoder
        images_norm_resize = tf.image.resize_images(
            images_norm, (gen_image_size, gen_image_size)
        )

        # Create encoder network
        mean, log_variance = vae.encoder(images_norm_resize, True)

        epsilon = tf.random_normal((tf.shape(mean)[0], args.latent_var_size))
        std = tf.exp(log_variance / 2)
        latent_var = mean + epsilon * std

        # Create decoder
        reconstructed_norm = vae.decoder(latent_var, False)

        # Un-normalize
        reconstructed = (reconstructed_norm * img_stddev) + img_mean

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Start running operations on the Graph
        gpu_memory_fraction = 1.0
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction
        )
        sess = tf.Session(
            config=tf.ConfigProto(
                gpu_options=gpu_options,
                log_device_placement=False
            )
        )
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # coord = tf.train.Coordinator()
        # tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if vae_checkpoint:
                print('Restoring VAE checkpoint: %s' % vae_checkpoint)
                saver.restore(sess, vae_checkpoint)

            start_time = time.time()
            latent_var_ = sess.run(latent_var, feed_dict={images: image})
            duration = time.time() - start_time
            print('Sess run: %.3f seconds' % duration)
            # NOTE: This will print the 'Out of range' warning if the last
            # batch is not full, as described by
            # https://github.com/tensorflow/tensorflow/issues/8330

            print('Got latent var shape {}'.format(latent_var_.shape))
            # Reconstruct faces while adding varying amount of
            # the selected attribute vector
            attribute_index = 31  # 31: 'Smiling'
            with h5py.File(args.attributes_filename, 'r') as f:
                # latent_vars = np.array(f.get('latent_vars'))
                # attributes = np.array(f.get('attributes'))
                # fields = np.array(f.get('fields'))
                attribute_vectors = np.array(f.get('attribute_vectors'))

            nrof_interp_steps = 10
            sweep_latent_var = np.zeros(
                (nrof_interp_steps, latent_var.shape[1]), np.float32
            )
            for i in range(nrof_interp_steps):
                sweep_latent_var[i, :] = (
                    latent_var_[0, :] + 5.0 * i /
                    nrof_interp_steps * attribute_vectors[attribute_index, :]
                )

            print('Reconstruct image')
            start_time = time.time()
            recon = sess.run(
                reconstructed,
                feed_dict={latent_var: sweep_latent_var}
            )
            duration = time.time() - start_time
            print('Sess run: %.3f seconds' % duration)

            img = facenet.put_images_on_grid(
                recon, shape=(nrof_interp_steps, int(math.ceil(1 / 2)))
            )

            image_filename = os.path.expanduser(args.output_image_filename)
            print('Writing generated image to %s' % image_filename)
            misc.imsave(image_filename, img)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'vae_def',
        type=str,
        help=(
            'Model definition for the variational autoencoder. Points to a '
            'module containing the definition.'
        )
    )
    parser.add_argument(
        'vae_checkpoint',
        type=str,
        help='Checkpoint file of a pre-trained variational autoencoder.'
    )
    parser.add_argument(
        'attributes_filename',
        type=str,
        help=(
            'The file containing the attribute vectors, as generated by '
            'calculate_attribute_vectors.py.'
        )
    )
    parser.add_argument(
        'output_image_filename',
        type=str,
        help='File to write the generated image to.'
    )
    parser.add_argument(
        '--image',
        help='Image filename to use.',
        required=True
    )
    parser.add_argument(
        '--latent_var_size',
        type=int,
        help='Dimensionality of the latent variable.',
        default=100
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed.',
        default=666
    )

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
