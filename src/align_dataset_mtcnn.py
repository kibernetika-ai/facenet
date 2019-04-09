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

import hashlib
import os
import pickle
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf
from ml_serving.drivers import driver

import base_args
import bg_remove
import facenet
import openvino_detection

# tf.logging.set_verbosity(tf.logging.INFO)
LOG = tf.logging


def print_fun(s):
    print(s)
    sys.stdout.flush()


def main(args):

    output_dir = os.path.expanduser(args.output_dir)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes.txt')
    align_filename = os.path.join(output_dir, 'align.pkl')

    align_data_args = dict(vars(args))
    # the next arguments can be changed w/o changing aligned images
    del align_data_args['complementary']
    del align_data_args['input_dir']
    del align_data_args['output_dir']

    align_data = {}
    clear_output_dir = True
    if args.complementary:
        if os.path.isfile(align_filename):
            print_fun("Check previous align data")
            with open(align_filename, 'rb') as infile:
                (align_data_args_loaded, align_data_loaded) = pickle.load(infile)
                if align_data_args == align_data_args_loaded:
                    print_fun("Loaded data about %d aligned classes" % len(align_data_loaded))
                    align_data = align_data_loaded
                    clear_output_dir = False
                else:
                    print_fun("Previous align data is for another arguments, skipped")

    if clear_output_dir:
        print_fun("Clearing output dir")
        shutil.rmtree(output_dir, ignore_errors=True)

    if not os.path.isdir(output_dir):
        print_fun("Creating output dir")
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
    serving.load_model(
        args.face_detection_path,
        device="CPU",
        flexible_batch_size=True,
    )

    bg_rm_drv = bg_remove.get_driver(args.bg_remove_path)

    input_name = list(serving.inputs.keys())[0]
    output_name = list(serving.outputs.keys())[0]

    threshold = 0.5

    min_face_area = args.min_face_size ** 2

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            output_class_dir_created = False
            if cls.name in align_data:
                align_data_class = align_data[cls.name]
            else:
                align_data_class = {}
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                if not os.path.exists(output_filename):
                    try:
                        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
                    except Exception as e:
                        error_message = '{}: {}'.format(image_path, e)
                        print_fun('ERROR: %s' % error_message)
                        continue

                    img_hash = hashlib.sha1(img.tostring()).hexdigest()
                    if image_path in align_data_class and align_data_class[image_path] == img_hash:
                        print_fun("%s - cached" % image_path)
                        continue
                    align_data_class[image_path] = hashlib.sha1(img.tostring()).hexdigest()
                    print_fun(image_path)

                    if len(img.shape) <= 2:
                        print_fun('WARNING: Unable to align "%s", shape %s' % (image_path, img.shape))
                        text_file.write('%s\n' % output_filename)
                        continue

                    if bg_rm_drv is not None:
                        img = bg_rm_drv.apply_mask(img)

                    serving_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
                    serving_img = np.transpose(serving_img, [2, 0, 1]).reshape([1, 3, 300, 300])
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

                    # Get the biggest box: find the box with largest square:
                    # (y1 - y0) * (x1 - x0) - size of box.
                    bbs = bounding_boxes
                    area = (bbs[:, 3] - bbs[:, 1]) * (bbs[:, 2] - bbs[:, 0])

                    if len(area) < 1:
                        print_fun('WARNING: Unable to align "%s", n_faces=%s' % (image_path, len(area)))
                        text_file.write('%s\n' % output_filename)
                        continue

                    num = np.argmax(area)
                    if area[num] < min_face_area:
                        print_fun(
                            'WARNING: Face found but too small - about {}px '
                            'width against required minimum of {}px. Try'
                            ' adjust parameter --min-face-size'.format(
                                int(np.sqrt(area[num])), args.min_face_size
                            )
                        )
                        continue

                    bounding_boxes = np.stack([bbs[num]])

                    imgs = openvino_detection.get_images(
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
                        if not output_class_dir_created:
                            output_class_dir_created = True
                            if not os.path.exists(output_class_dir):
                                os.makedirs(output_class_dir)
                        cv2.imwrite(output_filename_n, cropped)

            align_data[cls.name] = align_data_class

    with open(align_filename, 'wb') as align_file:
        pickle.dump((align_data_args, align_data), align_file, protocol=2)

    print_fun('Total number of images: %d' % nrof_images_total)
    print_fun('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    build_id = os.environ.get('BUILD_ID', None)
    if os.environ.get('PROJECT_ID', None) and (build_id is not None):
        from mlboardclient.api import client
        client.update_task_info({'aligned_location': output_dir})


def parse_arguments(argv):
    parser = base_args.parser()
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
        '--min-face-size',
        type=int,
        help='Minimum face size in pixels.',
        default=25
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
        '--complementary',
        help='Сomplementary training, existing aligned images in output dir supplements with new ones from input dir.',
        action='store_true'
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
