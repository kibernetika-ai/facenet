import openvino_detection
import base_args


def parser(description=None):
    p = base_args.parser(description)
    p.add_argument(
        '--classifier',
        nargs="*",
        help='Path to classifier files.',
    )
    p.add_argument(
        '--device',
        help='Device for openVINO.',
        default="CPU",
        choices=["CPU", "MYRIAD"]
    )
    p.add_argument(
        '--graph',
        help='Path to facenet openVINO graph.',
        default='facenet.xml',
    )
    p.add_argument(
        '--debug',
        help='Full debug output for each detected face.',
        action='store_true',
    )
    return p


def OpenVINOFacenet(args):
    return openvino_detection.OpenVINOFacenet(
        device=args.device,
        face_detection_path=args.face_detection_path,
        facenet_path=args.graph,
        classifier=args.classifier,
        bg_remove_path=args.bg_remove_path,
        debug=args.debug,
    )
