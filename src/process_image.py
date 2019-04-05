import argparse
import glob
import logging
import os

import cv2
import numpy as np
from openvino import inference_engine as ie

import openvino_detection


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Test movidius'
    )
    parser.add_argument(
        '--face-detection-path',
        default=None,
        help='Path to face-detection-retail openvino model',
        required=True,
    )
    parser.add_argument(
        '--classifier',
        help='Path to classifier file.',
    )
    parser.add_argument(
        '--device',
        help='Device for openVINO.',
        default="CPU",
        choices=["CPU", "MYRIAD"]
    )
    parser.add_argument(
        '--images',
        nargs='*',
        help='Path to the source image files to be processed (can be mask).'
             'Output image\'s name will be named as input image with prefix \'processed_\'',
    )
    parser.add_argument(
        '--graph',
        help='Path to facenet openVINO graph.',
        default='facenet.xml',
    )
    parser.add_argument(
        '--bg-remove-path',
        help='Path to Tensorflow background remove model.',
        default=None,
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if len(args.images) == 0:
        print("No input images specified")
        return

    facenet = openvino_detection.OpenVINOFacenet(
        args.device,
        args.face_detection_path,
        args.graph,
        args.classifier,
        args.bg_remove_path,
    )

    for img in args.images:
        try:
            img_dir, img_filename = os.path.split(img)
            img_name, img_extension = os.path.splitext(img_filename)
            output = os.path.join(img_dir, "processed_%s%s" % (img_name, img_extension))
            try:
                os.remove(output)
            except:
                pass
            image = cv2.imread(img, cv2.IMREAD_COLOR).astype(np.float32)
            facenet.process_frame(image)
            cv2.imwrite(output, image)
            print("Image %s processed and saved to %s" % (img, output))
        except Exception as e:
            print("Image %s process error: %s" % (img, e))
            pass


if __name__ == "__main__":
    main()
