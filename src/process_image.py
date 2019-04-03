import argparse
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
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for detecting faces',
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
        '--image',
        help='Path to the source image file to be processed.',
    )
    parser.add_argument(
        '--output',
        help='Path to the output (processed) image file to write to.',
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

    facenet = openvino_detection.OpenVINOFacenet(
        args.device,
        args.face_detection_path,
        args.graph,
        args.classifier,
        args.bg_remove_path,
    )

    image = cv2.imread(args.image, cv2.IMREAD_COLOR).astype(np.float32)
    try:
        os.remove(args.output)
    except:
        pass
    facenet.process_frame(image)
    cv2.imwrite(args.output, image)


if __name__ == "__main__":
    main()
