"""Performs face alignment and stores face thumbnails in the output directory."""
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
import os
import random
import sys

import cv2
from ml_serving.drivers import driver
import numpy as np
import tensorflow as tf

import camera_openvino as ko
import facenet

# tf.logging.set_verbosity(tf.logging.INFO)
LOG = tf.logging


def print_fun(s):
    print(s)
    sys.stdout.flush()


def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print_fun('Creating networks and loading parameters')

    # Load driver
    drv = driver.load_driver("openvino")
    # Instantinate driver
    serving = drv()
    model = (
        "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/"
        "face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
    )
    serving.load_model(
        model,
        device="CPU",
        flexible_batch_size=True,
    )
    input_name = list(serving.inputs.keys())[0]
    output_name = list(serving.outputs.keys())[0]

    threshold = 0.5

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print_fun(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
                    except Exception as e:
                        error_message = '{}: {}'.format(image_path, e)
                        print_fun('ERROR: %s' % error_message)
                        continue

                    if len(img.shape) <= 2:
                        print_fun('WARNING: Unable to align "%s", shape %s' % (image_path, img.shape))
                        text_file.write('%s\n' % output_filename)
                        continue

                    serving_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
                    serving_img = np.transpose(serving_img, [2, 0, 1]).reshape(1, 3, 300, 300)
                    raw = serving.predict({input_name: serving_img})[output_name].reshape([-1, 7])
                    # 7 values:
                    # class_id, label, confidence, x_min, y_min, x_max, y_max
                    # Select boxes where confidence > factor
                    bboxes_raw = raw[raw[:, 2] > threshold]
                    bboxes_raw[:, 3] = bboxes_raw[:, 3] * img.shape[1]
                    bboxes_raw[:, 5] = bboxes_raw[:, 5] * img.shape[1]
                    bboxes_raw[:, 4] = bboxes_raw[:, 4] * img.shape[0]
                    bboxes_raw[:, 6] = bboxes_raw[:, 6] * img.shape[0]

                    bounding_boxes = np.zeros([len(bboxes_raw), 5])

                    bounding_boxes[:, 0:4] = bboxes_raw[:, 3:7]
                    bounding_boxes[:, 4] = bboxes_raw[:, 2]

                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces < 1:
                        print_fun('WARNING: Unable to align "%s", n_faces=%s' % (image_path, nrof_faces))
                        text_file.write('%s\n' % output_filename)

                    imgs = ko.get_images(
                        img,
                        bounding_boxes,
                        face_crop_size=args.image_size,
                        face_crop_margin=args.margin,
                        prewhiten=False,
                    )
                    for i, cropped in enumerate(imgs):
                        nrof_successfully_aligned += 1
                        bb = bounding_boxes[i]
                        filename_base, file_extension = os.path.splitext(output_filename)
                        output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)

                        text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        cv2.imwrite(output_filename_n, cropped)

    print_fun('Total number of images: %d' % nrof_images_total)
    print_fun('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    build_id = os.environ.get('BUILD_ID', None)
    if os.environ.get('PROJECT_ID', None) and (build_id is not None):
        from mlboardclient.api import client
        client.update_task_info({'aligned_location': output_dir})


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory with unaligned images.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory with aligned face thumbnails.'
    )
    parser.add_argument(
        '--model_dir',
        type=str, default=None,
        help='Model location'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        help='Image size (height, width) in pixels.',
        default=160
    )
    parser.add_argument(
        '--margin',
        type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.',
        default=32
    )
    parser.add_argument(
        '--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.',
        action='store_true'
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
