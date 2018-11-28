import argparse
import logging
import os
import pickle
import time

import cv2
import numpy as np
from openvino import inference_engine as ie
from scipy import misc
import six

import align.detect_face as detect_face
import facenet


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
        '--factor',
        type=float,
        default=0.709,
        help='Factor',
    )
    parser.add_argument(
        '--resolutions',
        type=str,
        default="26x37,37x52,52x74,145x206",
        help='PNET resolutions',
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


def get_size(scale):
    t = scale.split('x')
    return int(t[0]), int(t[1])


def imresample(img, h, w):
    im_data = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)  # @UndefinedVariable
    return im_data


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
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0),
                thickness=1, lineType=2
            )


def parse_resolutions(v):
    res = []
    for r in v.split(','):
        hw = r.split('x')
        if len(hw) == 2:
            res.append((int(hw[0]), int(hw[1])))
    return res


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
            scaled = misc.imresize(cropped, (face_crop_size, face_crop_size), interp='bilinear')
            images.append(facenet.prewhiten(scaled))

    return images


class PNetHandler(object):
    def __init__(self, plugin, h, w, net_dir=None):
        net = ie.IENetwork.from_ir(*net_filenames(
            plugin, 'pnet_{}x{}'.format(h, w), net_dir=net_dir)
        )
        self.input_name = list(net.inputs.keys())[0]
        LOG.info(net.outputs)
        outputs = list(iter(net.outputs))
        self.output_name0 = outputs[0]
        self.output_name1 = outputs[1]
        self.exec_net = plugin.load(net)
        self.h = h
        self.w = w

    def destroy(self):
        pass

    def proxy(self):
        def _exec(img):
            # Channel first
            img = img.transpose([0, 3, 1, 2])
            output = self.exec_net.infer({self.input_name: img})
            output1 = output[self.output_name0]
            output2 = output[self.output_name1]
            # Channel last
            output1 = output1.transpose([0, 3, 2, 1])
            output2 = output2.transpose([0, 3, 2, 1])
            return output1, output2

        return _exec, self.h, self.w


def net_filenames(plugin, net_name, net_dir=None):
    if not net_dir:
        device = plugin.device.lower()
        net_dir = 'openvino-{}'.format(device)
    base_name = '{}/{}'.format(net_dir, net_name)
    xml_name = base_name + '.xml'
    bin_name = base_name + '.bin'
    return xml_name, bin_name


def main():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
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

    print('Load PNET')

    pnets = []
    for r in parse_resolutions(args.resolutions):
        p = PNetHandler(plugin, r[0], r[1])
        pnets.append(p)

    print('Load RNET')
    net = ie.IENetwork.from_ir(*net_filenames(plugin, 'rnet'))
    rnet_input_name = list(net.inputs.keys())[0]
    rnet_output_name0 = net.outputs[0]
    rnet_output_name1 = net.outputs[1]
    r_net = plugin.load(net)

    print('Load ONET')

    net = ie.IENetwork.from_ir(*net_filenames(plugin, 'onet'))
    onet_input_name = list(net.inputs.keys())[0]
    onet_batch_size = net.inputs[onet_input_name][0]
    onet_output_name0 = net.outputs[0]
    onet_output_name1 = net.outputs[1]
    onet_output_name2 = net.outputs[2]
    print('ONET_BATCH_SIZE = {}'.format(onet_batch_size))
    o_net = plugin.load(net)

    if use_classifier:
        print('Load FACENET')

        model_file = args.graph
        weights_file = model_file[:-3] + 'bin'

        net = ie.IENetwork.from_ir(model_file, weights_file)
        facenet_input = list(net.inputs.keys())[0]
        facenet_output = net.outputs[0]
        face_net = plugin.load(net)

        # Load classifier
        with open(args.classifier, 'rb') as f:
            opts = {'file': f}
            if six.PY3:
                opts['encoding'] = 'latin1'
            (model, class_names) = pickle.load(**opts)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # video_capture = cv2.VideoCapture(0)
    if args.image is None:
        from imutils.video import VideoStream
        from imutils.video import FPS
        vs = VideoStream(
            usePiCamera=args.camera_device == "PI",
            # resolution=(640, 480),
            # framerate=24
        ).start()
        time.sleep(1)
        fps = FPS().start()

    bounding_boxes = []
    labels = []

    pnets_proxy = []
    for p in pnets:
        pnets_proxy.append(p.proxy())

    def _rnet_proxy(img):
        output = r_net.infer({rnet_input_name: img})
        return output[rnet_output_name0], output[rnet_output_name1]

    def _onet_proxy(img):
        # img = img.reshape([1, 3, 48, 48])
        output = o_net.infer({onet_input_name: img})
        return output[onet_output_name0], output[onet_output_name1], output[onet_output_name2]

    pnet, rnet, onet = detect_face.create_openvino_mtcnn(
        pnets_proxy, _rnet_proxy, _onet_proxy, onet_batch_size
    )
    try:
        while True:
            # Capture frame-by-frame
            if args.image is None:
                frame = vs.read()
                if isinstance(frame, tuple):
                    frame = frame[1]
            else:
                frame = cv2.imread(args.image).astype(np.float32)

            # h = 400
            # w = int(round(frame.shape[1] / (frame.shape[0] / float(h))))
            h = 480
            w = 640
            if (frame.shape[1] != w) or (frame.shape[0] != h):
                frame = cv2.resize(
                    frame, (w, h), interpolation=cv2.INTER_AREA
                )

            # BGR -> RGB
            rgb_frame = frame[:, :, ::-1]
            # rgb_frame = frame
            # print("Frame {}".format(frame.shape))

            if (frame_count % frame_interval) == 0:
                # t = time.time()
                bounding_boxes, _ = detect_face.detect_face_openvino(
                    rgb_frame, pnet, rnet, onet, threshold
                )
                # d = (time.time() - t) * 1000
                # LOG.info('recognition: %.3fms' % d)
                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count/(end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

                if use_classifier:
                    imgs = get_images(rgb_frame, bounding_boxes)
                    labels = []
                    # t = time.time()
                    for img_idx, img in enumerate(imgs):
                        img = img.astype(np.float32)

                        # Infer
                        img = img.transpose([2, 0, 1]).reshape([1, 3, 160, 160])
                        output = face_net.infer({facenet_input: img})
                        output = output[facenet_output]
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
                    # d = (time.time() - t) * 1000
                    # LOG.info('facenet: %.3fms' % d)

            add_overlays(frame, bounding_boxes, frame_rate, labels=labels)

            frame_count += 1
            if args.image is None:
                cv2.imshow('Video', frame)
            else:
                print(bounding_boxes)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except (KeyboardInterrupt, SystemExit) as e:
        print('Caught %s: %s' % (e.__class__.__name__, e))

    # When everything is done, release the capture
    # video_capture.release()
    if args.image is None:
        fps.stop()
        vs.stop()
        cv2.destroyAllWindows()
    print('Finished')


if __name__ == "__main__":
    main()
