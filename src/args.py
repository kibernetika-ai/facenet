import argparse


def base_parser(description=None):
    parser = argparse.ArgumentParser(
        description=description,
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
