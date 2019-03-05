import argparse
import logging
import os
import pickle
import time

import cv2
import numpy as np
from openvino import inference_engine as ie
import six

import openvino_nets as nets
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


def add_overlays(frame, boxes, frame_rate, labels=None):
    if boxes is not None:
        for face in boxes:
            face_bb = face.astype(int)
            cv2.rectangle(
                frame,
                (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                (0, 255, 0), 2
            )

    if frame_rate != 0:
        cv2.putText(
            frame, str(frame_rate) + " fps", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            thickness=2, lineType=2
        )

    if labels:
        for l in labels:
            cv2.putText(
                frame, l['label'], (l['left'], l['top'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0),
                thickness=1, lineType=cv2.LINE_AA
            )


def get_images(image, bounding_boxes):
    face_crop_size = 160
    face_crop_margin = 32
    images = []

    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(image.shape)[0:2]
        if nrof_faces > 1:
            for i in range(nrof_faces):
                det_arr.append(np.squeeze(det[i]))
        else:
            det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - face_crop_margin / 2, 0)
            bb[1] = np.maximum(det[1] - face_crop_margin / 2, 0)
            bb[2] = np.minimum(det[2] + face_crop_margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + face_crop_margin / 2, img_size[0])
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cv2.resize(cropped, (face_crop_size, face_crop_size), interpolation=cv2.INTER_AREA)
            images.append(utils.prewhiten(scaled))

    return images


def openvino_detect(face_detect, frame, threshold):
    inference_frame = cv2.resize(frame, face_detect.input_size, interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(*face_detect.input_shape)
    outputs = face_detect(inference_frame)
    outputs = outputs.reshape(-1, 7)
    bboxes_raw = outputs[outputs[:, 2] > threshold]
    bounding_boxes = bboxes_raw[:, 3:7]
    bounding_boxes[:, 0] = bounding_boxes[:, 0] * frame.shape[1]
    bounding_boxes[:, 2] = bounding_boxes[:, 2] * frame.shape[1]
    bounding_boxes[:, 1] = bounding_boxes[:, 1] * frame.shape[0]
    bounding_boxes[:, 3] = bounding_boxes[:, 3] * frame.shape[0]

    return bounding_boxes


def main():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 3  # seconds
    frame_rate = 0
    frame_count = 0
    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()

    use_classifier = bool(args.classifier)

    extensions = os.environ.get('INTEL_EXTENSIONS_PATH')
    plugin = ie.IEPlugin(device=args.device)

    if extensions and "CPU" in args.device:
        for ext in extensions.split(':'):
            print("LOAD extension from {}".format(ext))
            plugin.add_cpu_extension(ext)

    print('Load FACE DETECTION')
    face_path = args.face_detection_path
    weights_file = face_path[:face_path.rfind('.')] + '.bin'
    net = ie.IENetwork.from_ir(face_path, weights_file)
    face_detect = nets.FaceDetect(plugin, net)

    if use_classifier:
        print('Load FACENET')

        model_file = args.graph
        weights_file = model_file[:model_file.rfind('.')] + '.bin'

        net = ie.IENetwork.from_ir(model_file, weights_file)
        facenet_input = list(net.inputs.keys())[0]
        outputs = list(iter(net.outputs))
        facenet_output = outputs[0]
        face_net = plugin.load(net, num_requests=2)

        # Load classifier
        with open(args.classifier, 'rb') as f:
            opts = {'file': f}
            if six.PY3:
                opts['encoding'] = 'latin1'
            (model, class_names) = pickle.load(**opts)

    # video_capture = cv2.VideoCapture(0)
    if args.image is None:
        from imutils.video import VideoStream
        vs = VideoStream(
            usePiCamera=args.camera_device == "PI",
            resolution=(640, 480),
            # framerate=24
        ).start()
        if args.camera_device == "PI":
            time.sleep(1)

    bounding_boxes = []
    labels = []

    cur_request = 0
    next_request = 1

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

    frame = get_frame()
    if frame is None:
        vs.stop()
        if hasattr(vs, 'stream') and hasattr(vs.stream, 'release'):
            vs.stream.release()
        return

    try:
        while True:
            # Capture frame-by-frame
            next_frame = get_frame()
            if next_frame is None:
                break
            # BGR -> RGB
            # rgb_frame = frame[:, :, ::-1]
            # rgb_frame = frame

            if (frame_count % frame_interval) == 0:
                # t = time.time()
                bounding_boxes = openvino_detect(face_detect, next_frame, args.threshold)
                # d = (time.time() - t) * 1000
                # LOG.info('recognition: %.3fms' % d)
                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count/(end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

                if use_classifier:
                    # t = time.time()
                    imgs = get_images(next_frame, bounding_boxes)
                    labels = []
                    for img_idx, img in enumerate(imgs):
                        #img = img.astype(np.float32)

                        # Infer
                        # t = time.time()
                        img = img.transpose([2, 0, 1]).reshape([1, 3, 160, 160])
                        face_net.start_async(request_id=next_request, inputs={facenet_input: img})

                    if face_net.requests[cur_request].wait(-1) == 0:
                        output = face_net.requests[cur_request].outputs[facenet_output]
                        # output = face_net.infer({facenet_input: img})
                        # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))
                        # output = output[facenet_output]
                        try:
                            output = output.reshape(1, model.shape_fit_[1])
                            predictions = model.predict_proba(output)
                        except ValueError as e:
                            # Can not reshape
                            print(
                                "ERROR: Output from graph doesn't consistent"
                                " with classifier model: %s" % e
                            )
                            continue

                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)),
                            best_class_indices
                        ]

                        for i in range(len(best_class_indices)):
                            bb = bounding_boxes[img_idx].astype(int)
                            text = '%.1f%% %s' % (
                                best_class_probabilities[i] * 100,
                                class_names[best_class_indices[i]]
                            )
                            labels.append({
                                'label': text,
                                'left': bb[0],
                                'top': bb[1] - 5
                            })
                            # DEBUG
                            print('%4d  %s: %.3f' % (
                                i,
                                class_names[best_class_indices[i]],
                                best_class_probabilities[i])
                            )
                    # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))

            add_overlays(frame, bounding_boxes, frame_rate, labels=labels)

            frame_count += 1
            if args.image is None:
                cv2.imshow('Video', frame)
            else:
                print(bounding_boxes)
                break

            next_request, cur_request = cur_request, next_request
            frame = next_frame

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
