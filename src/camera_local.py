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
import utils


def get_parser():
    parser = argparse.ArgumentParser(
        description='Test movidious'
    )
    parser.add_argument(
        '--size',
        default='360x480',
        help='Image size',
    )
    parser.add_argument(
        '--image',
        help='Image',
    )
    parser.add_argument(
        '--align-dir',
        help='Path to the tensorflow models align dir',
        required=True,
    )
    parser.add_argument(
        '--classifier',
        help='Path to classifier file.',
    )
    parser.add_argument(
        '--camera',
        help='Full URL to network camera.',
    )
    parser.add_argument('--tf-graph-path')
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


def main():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    if bool(args.classifier) ^ bool(args.tf_graph_path):
        raise ValueError('tf_graph path and classifier must be filled.')

    use_classifier = args.classifier and args.tf_graph_path

    if args.camera:
        video_capture = cv2.VideoCapture(args.camera)
    else:
        video_capture = cv2.VideoCapture(0)

    bounding_boxes = []
    labels = []

    with tf.Session() as sess:
        pnet, rnet, onet = detect_face.create_mtcnn(sess, args.align_dir)
        if use_classifier:
            # Load classifier
            with open(args.classifier, 'rb') as f:
                opts = {'file': f}
                if six.PY3:
                    opts['encoding'] = 'latin1'
                (model, class_names) = pickle.load(**opts)

            facenet.load_model(args.tf_graph_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        try:
            while True:
                if not args.image:
                    _, frame = video_capture.read()
                else:
                    frame = cv2.imread(args.image)
                frame = utils.image_resize(frame, height=480)
                # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                # BGR -> RGB
                rgb_frame = frame[:, :, ::-1]

                if (frame_count % frame_interval) == 0:
                    bounding_boxes, _ = detect_face.detect_face(
                        rgb_frame, minsize, pnet, rnet, onet, threshold, factor
                    )
                    # Check our current fps
                    end_time = time.time()
                    if (end_time - start_time) > fps_display_interval:
                        frame_rate = int(frame_count / (end_time - start_time))
                        start_time = time.time()
                        frame_count = 0

                    if use_classifier:
                        imgs = get_images(rgb_frame, bounding_boxes)
                        labels = []
                        for img_idx, img in enumerate(imgs):
                            img = img.astype(np.float32)
                            feed_dict = {
                                images_placeholder: [img],
                                phase_train_placeholder: False
                            }
                            embedding = sess.run(embeddings, feed_dict=feed_dict)
                            try:
                                predictions = model.predict_proba(embedding)
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
                                # print('%4d  %s: %.3f' % (
                                #     i,
                                #     class_names[best_class_indices[i]],
                                #     best_class_probabilities[i])
                                # )

                add_overlays(frame, bounding_boxes, frame_rate, labels=labels)

                frame_count += 1
                cv2.imshow('Video', frame)

                key = cv2.waitKey(1)
                # Wait 'q' or Esc
                if key == ord('q') or key == 27:
                    break

        except (KeyboardInterrupt, SystemExit) as e:
            print('Caught %s: %s' % (e.__class__.__name__, e))

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print('Finished')


if __name__ == "__main__":
    main()
