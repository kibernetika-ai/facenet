import argparse
import bg_remove


def parser(description=None):
    p = argparse.ArgumentParser(
        description=description,
    )
    p.add_argument(
        '--face-detection-path',
        default='/opt/intel/computer_vision_sdk/deployment_tools/intel_models/'
                'face-detection-retail-0004/FP32/face-detection-retail-0004.xml',
        help='Path to face-detection-retail openvino model',
        required=True,
    )
    bg_remove.add_bg_remove_arg(p)
    return p