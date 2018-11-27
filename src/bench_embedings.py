"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import sys

import numpy as np
import time

import facenet
from ml_serving.drivers import driver





def main(args):


    dataset = facenet.get_dataset(args.data_dir)
    # Check that there are at least one training image per class
    for cls in dataset:
        assert len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset'

    paths, labels = facenet.get_image_paths_and_labels(dataset)

    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    # Load the model
    print('Loading feature extraction model')

    # Load driver
    drv = driver.load_driver(args.driver)
    # Instantinate driver
    serving = drv()
    serving.load_model(
        args.model,
        inputs='input:0,phase_train:0',
        outputs='embeddings:0',
        device=args.device,
        flexible_batch_size=True,
    )

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    nrof_images = len(paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
    embeddings_size = nrof_images
    emb_array = np.zeros((embeddings_size, 512))
    start_time = time.time()
    for i in range(nrof_batches_per_epoch):
        start_index = i * args.batch_size
        end_index = min((i + 1) * args.batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, args.image_size)

        if serving.driver_name == 'tensorflow':
            feed_dict = {'input:0': images, 'phase_train:0': False}
        elif serving.driver_name == 'openvino':
            input_name = list(serving.inputs.keys())[0]

            # Transpose image for channel first format
            images = images.transpose([0, 3, 1, 2])
            feed_dict = {input_name: images}
        else:
            raise RuntimeError('Driver %s currently not supported' % serving.driver_name)

        outputs = serving.predict(feed_dict)

    end_time = time.time()
    print("Duration: {} sec/sample batch count:{}".format((end_time-start_time)/nrof_batches_per_epoch,nrof_batches_per_epoch))




def parse_arguments(argv):
    parser = argparse.ArgumentParser()


    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the data directory containing aligned LFW face patches.'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to .xml openVINO IR file',
        required=True,
    )

    parser.add_argument(
        '--device',
        help='Device for openVINO.',
        default="CPU",
        choices=["CPU", "MYRIAD"]
    )
    parser.add_argument(
        '--driver',
        help='Driver for inference.',
        default="tensorflow",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Number of images to process in a batch.',
        default=1
    )

    parser.add_argument(
        '--image_size',
        type=int,
        help='Image size (height, width) in pixels.',
        default=160
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
