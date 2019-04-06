import argparse
import openvino_detection


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
        nargs="*",
        help='Path to classifier files.',
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
    parser.add_argument(
        '--debug',
        help='Full debug output for each detected face.',
        action='store_true',
    )
    return parser


def OpenVINOFacenet(args):
    return openvino_detection.OpenVINOFacenet(
        device=args.device,
        face_detection_path=args.face_detection_path,
        facenet_path=args.graph,
        classifier=args.classifier,
        bg_remove_path=args.bg_remove_path,
        debug=args.debug,
    )
