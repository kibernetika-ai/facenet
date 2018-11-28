from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import sys

from ml_serving.drivers import driver
import numpy as np
import time

import facenet
import serving_hook


class Context(object):
    pass


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
    serving = drv(
        preprocess=serving_hook.preprocess,
        postprocess=serving_hook.postprocess,
        init_hook=serving_hook.init_hook,
        classifier=args.classifier,
        use_tf='False',
        use_face_detection='True',
        face_detection_path=args.face_detection_path
    )
    serving.load_model(
        args.model,
        inputs='input:0,phase_train:0',
        outputs='embeddings:0',
        device=args.device,
        flexible_batch_size=True,
    )

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    time_requests = 0.0
    epochs = 2
    start_time = time.time()
    for j in range(epochs):
        for path in paths:
            print('Processing %s...' % path)
            with open(path, 'rb') as f:
                data = f.read()

            t = time.time()

            feed_dict = {'input': np.array(data)}
            outputs = serving.predict_hooks(feed_dict, context=Context())

            delta = (time.time() - t) * 1000
            time_requests += delta

    duration = float(time.time() - start_time)
    print()
    print('Total time: %.3fs' % duration)
    per_request_ms = float(time_requests) / epochs / len(paths)
    print('Time per request: %.3fms' % per_request_ms)

    speed = 1 / (per_request_ms / 1000)
    print('Speed: {} sample/sec'.format(speed))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the data directory containing images.'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to .xml openVINO IR file',
        required=True,
    )
    parser.add_argument(
        '--classifier',
        type=str,
        help='Path to classifier .pkl file',
        required=True,
    )
    parser.add_argument(
        '--face-detection-path',
        type=str,
        help='Path to face-detection .xml openVINO IR file',
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
