from mvnc import mvncapi as mvnc
import numpy as np
import pickle
import cv2
import argparse
import align.detect_face as detect_face
import tensorflow as tf
import numpy as np
import time
import six
from scipy import misc

import facenet


def get_parser():
    parser = argparse.ArgumentParser(
        description='Test movidious'
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
        default="37x52,73x104",
        help='PNET resolutions',
    )
    parser.add_argument(
        '--classifier',
        help='Path to classifier file.',
    )
    parser.add_argument(
        '--graph',
        help='Path to facenet graph.',
        default='facenet.graph',
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


def _mvc_exec(img, h, w, pnetGraph, pnetIn, pnetOut):
    # print("Exec {}x{} on {}".format(h, w, img.shape))
    pnetGraph.queue_inference_with_fifo_elem(pnetIn, pnetOut, img, 'pnet')
    output, userobj = pnetOut.read_elem()
    return output


class PNetHandler(object):
    def __init__(self, device, h, w):
        with open('movidius/pnet-{}x{}.graph'.format(h, w), mode='rb') as f:
            graphFileBuff = f.read()
        self.pnetGraph = mvnc.Graph('PNet Graph {}x{}'.format(h, w))
        self.pnetIn, self.pnetOut = self.pnetGraph.allocate_with_fifos(device, graphFileBuff)
        self.h = h
        self.w = w

    def destroy(self):
        self.pnetIn.destroy()
        self.pnetOut.destroy()
        self.pnetGraph.destroy()

    def proxy(self):
        f = (lambda x: _mvc_exec(x, self.h, self.w, self.pnetGraph, self.pnetIn, self.pnetOut))
        return f, self.h, self.w


def main():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()

    use_classifier = bool(args.classifier)

    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    device = mvnc.Device(devices[0])
    device.open()

    print('Load PNET')

    pnets = []
    for r in parse_resolutions(args.resolutions):
        p = PNetHandler(device, r[0], r[1])
        pnets.append(p)

    print('Load RNET')

    with open('movidius/rnet.graph', mode='rb') as f:
        rgraphFileBuff = f.read()
    rnetGraph = mvnc.Graph("RNet Graph")
    rnetIn, rnetOut = rnetGraph.allocate_with_fifos(device, rgraphFileBuff)

    print('Load ONET')

    with open('movidius/onet.graph', mode='rb') as f:
        ographFileBuff = f.read()
    onetGraph = mvnc.Graph("ONet Graph")
    onetIn, onetOut = onetGraph.allocate_with_fifos(device, ographFileBuff)

    if use_classifier:
        print('Load FACENET')

        with open(args.graph, mode='rb') as f:
            fgraphFileBuff = f.read()
        fGraph = mvnc.Graph("Face Graph")
        fifoIn, fifoOut = fGraph.allocate_with_fifos(device, fgraphFileBuff)

        # Load classifier
        with open(args.classifier, 'rb') as f:
            opts = {'file': f}
            if six.PY3:
                opts['encoding'] = 'latin1'
            (model, class_names) = pickle.load(**opts)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.6, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # video_capture = cv2.VideoCapture(0)
    if args.image is None:
        from imutils.video import VideoStream
        from imutils.video import FPS
        vs = VideoStream(usePiCamera=True, resolution=(640, 480), framerate=24).start()
        time.sleep(1)
        fps = FPS().start()

    bounding_boxes = []
    labels = []

    with tf.Session() as sess:
        pnets_proxy = []
        for p in pnets:
            pnets_proxy.append(p.proxy())

        def _rnet_proxy(img):
            rnetGraph.queue_inference_with_fifo_elem(rnetIn, rnetOut, img, 'rnet')
            output, userobj = rnetOut.read_elem()
            return output

        def _onet_proxy(img):
            onetGraph.queue_inference_with_fifo_elem(onetIn, onetOut, img, 'onet')
            output, userobj = onetOut.read_elem()
            return output

        pnets_proxy, rnet, onet = detect_face.create_movidius_mtcnn(
            sess, 'align', pnets_proxy, _rnet_proxy, _onet_proxy
        )
        try:
            while True:
                # Capture frame-by-frame
                if args.image is None:
                    frame = vs.read()
                else:
                    frame = cv2.imread(args.image).astype(np.float32)

                if (frame.shape[1] != 640) or (frame.shape[0] != 480):
                    frame = cv2.resize(
                        frame, (640, 480), interpolation=cv2.INTER_AREA
                    )

                # BGR -> RGB
                rgb_frame = frame[:, :, ::-1]
                # print("Frame {}".format(frame.shape))

                if (frame_count % frame_interval) == 0:
                    bounding_boxes, _ = detect_face.movidius_detect_face(
                        rgb_frame, pnets_proxy, rnet, onet, threshold
                    )

                    # Check our current fps
                    end_time = time.time()
                    if (end_time - start_time) > fps_display_interval:
                        frame_rate = int(frame_count/(end_time - start_time))
                        start_time = time.time()
                        frame_count = 0

                if use_classifier:
                    imgs = get_images(rgb_frame, bounding_boxes)
                    labels = []
                    for img_idx, img in enumerate(imgs):
                        img = img.astype(np.float32)
                        fGraph.queue_inference_with_fifo_elem(
                            fifoIn, fifoOut, img, 'user object'
                        )
                        output, userobj = fifoOut.read_elem()
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

                        print(output.shape)
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
    if use_classifier:
        fifoIn.destroy()
        fifoOut.destroy()
        fGraph.destroy()
    rnetIn.destroy()
    rnetOut.destroy()
    rnetGraph.destroy()
    onetIn.destroy()
    onetOut.destroy()
    onetGraph.destroy()
    for p in pnets:
        p.destroy()
    device.close()
    print('Finished')


if __name__ == "__main__":
    main()
