import openvino_args
import logging
import os

import cv2
import numpy as np


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_parser():
    parser = openvino_args.parser('Test movidius')
    parser.add_argument(
        '--images',
        nargs='*',
        help='Path to the source image files to be processed (can be mask).'
             'Output image\'s name will be named as input image with prefix \'processed_\'',
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if len(args.images) == 0:
        print("No input images specified")
        return

    facenet = openvino_args.OpenVINOFacenet(args)

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
            print("Image %s processed" % img)
        except Exception as e:
            print("Image %s process error: %s" % (img, e))
            pass


if __name__ == "__main__":
    main()
