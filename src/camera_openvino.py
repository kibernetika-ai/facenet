import argparse
import logging
import time

import cv2
import numpy as np
from openvino import inference_engine as ie

import openvino_detection
import utils


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Test movidius'
    )
    parser.add_argument(
        '--image',
        default=None,
        help='Image',
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
        default="MYRIAD",
        choices=["CPU", "MYRIAD"]
    )
    parser.add_argument(
        '--camera-device',
        help='Lib for camera to use.',
        default="PI",
        choices=["PI", "CV"]
    )
    parser.add_argument(
        '--graph',
        help='Path to facenet openVINO graph.',
        default='facenet.xml',
    )
    return parser


def main():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 3  # seconds
    frame_rate = 0
    frame_count = 0
    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()

    facenet = openvino_detection.OpenVINOFacenet(
        args.device,
        args.face_detection_path,
        args.graph,
        args.classifier
    )

    # video_capture = cv2.VideoCapture(0)
    if args.image is None:
        from imutils.video import VideoStream
        vs = VideoStream(
            usePiCamera=args.camera_device == "PI",
            resolution=(640, 480),
            # framerate=24
        )
        if args.camera_device == "CV":
            default_w = vs.stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            default_h = vs.stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
            vs.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            vs.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, _ = vs.stream.stream.read()
            if not ret:
                # fallback
                vs.stream.stream.release()
                vs.stream.stream.open(0)
                vs.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, default_w)
                vs.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, default_h)

        vs.start()
        if args.camera_device == "PI":
            time.sleep(1)

    bounding_boxes = []
    labels = []

    def get_frame():
        if args.image is None:
            new_frame = vs.read()
            if isinstance(new_frame, tuple):
                new_frame = new_frame[1]
        else:
            new_frame = cv2.imread(args.image).astype(np.float32)

        if new_frame is None:
            print("frame is None. Possibly camera or display does not work")
            return None

        if new_frame.shape[0] > 480:
            new_frame = utils.image_resize(new_frame, height=480)

        return new_frame

    try:
        while True:
            # Capture frame-by-frame
            frame = get_frame()
            if frame is None:
                break
            # BGR -> RGB
            # rgb_frame = frame[:, :, ::-1]
            # rgb_frame = frame

            if (frame_count % frame_interval) == 0:
                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count/(end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

                facenet.process_frame(frame, args.threshold, frame_rate=frame_rate)

            frame_count += 1
            if args.image is None:
                cv2.imshow('Video', frame)
            else:
                print(bounding_boxes)
                print(labels)
                break

            key = cv2.waitKey(1)
            # Wait 'q' or Esc or 'q' in russian layout
            if key in [ord('q'), 202, 27]:
                break

    except (KeyboardInterrupt, SystemExit) as e:
        print('Caught %s: %s' % (e.__class__.__name__, e))

    # When everything is done, release the capture
    # video_capture.release()
    if args.image is None:
        vs.stop()
        if hasattr(vs, 'stream') and hasattr(vs.stream, 'release'):
            vs.stream.release()
        cv2.destroyAllWindows()
    print('Finished')

    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()
