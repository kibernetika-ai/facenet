import os
import pickle

import cv2
import numpy as np
from openvino import inference_engine as ie
import six
from sklearn import neighbors
from sklearn import svm

import openvino_nets as nets
import utils


class OpenVINOFacenet(object):
    def __init__(self, device, face_detection_path, facenet_path=None, classifier=None):
        self.use_classifier = False

        extensions = os.environ.get('INTEL_EXTENSIONS_PATH')
        plugin = ie.IEPlugin(device=device)

        if extensions and "CPU" in device:
            for ext in extensions.split(':'):
                print("LOAD extension from {}".format(ext))
                plugin.add_cpu_extension(ext)

        print('Load FACE DETECTION')
        face_path = face_detection_path
        weights_file = face_path[:face_path.rfind('.')] + '.bin'
        net = ie.IENetwork(face_path, weights_file)
        self.face_detect = nets.FaceDetect(plugin, net)

        if facenet_path and classifier:
            self.use_classifier = True
            print('Load FACENET')

            model_file = facenet_path
            weights_file = model_file[:model_file.rfind('.')] + '.bin'

            net = ie.IENetwork(model_file, weights_file)
            self.facenet_input = list(net.inputs.keys())[0]
            outputs = list(iter(net.outputs))
            self.facenet_output = outputs[0]
            self.face_net = plugin.load(net)

            # Load classifier
            with open(classifier, 'rb') as f:
                opts = {'file': f}
                if six.PY3:
                    opts['encoding'] = 'latin1'
                (model, class_names) = pickle.load(**opts)
                if isinstance(model, svm.SVC):
                    self.embedding_size = model.shape_fit_[1]
                elif isinstance(model, neighbors.KNeighborsClassifier):
                    self.embedding_size = model._fit_X.shape[1]
                else:
                    # try embedding_size = 512
                    self.embedding_size = 512
                self.classifier = model
                self.class_names = class_names

    def process_frame(self, frame, threshold=0.5, frame_rate=None):
        bounding_boxes = openvino_detect(self.face_detect, frame, threshold)

        labels = []
        if self.use_classifier:
            # t = time.time()
            imgs = get_images(frame, bounding_boxes)

            for img_idx, img in enumerate(imgs):
                # Infer
                # t = time.time()
                # Convert BGR to RGB
                img = img[:, :, ::-1]
                img = img.transpose([2, 0, 1]).reshape([1, 3, 160, 160])
                output = self.face_net.infer(inputs={self.facenet_input: img})

                output = output[self.facenet_output]
                # output = face_net.infer({facenet_input: img})
                # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))
                # output = output[facenet_output]
                try:
                    output = output.reshape(1, self.embedding_size)
                    predictions = self.classifier.predict_proba(output)
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
                        self.class_names[best_class_indices[i]]
                    )
                    labels.append({
                        'label': text,
                        'left': bb[0],
                        'top': bb[1] - 5,
                        'bottom': bb[3] + 7
                    })
                    # DEBUG
                    print('%4d  %s: %.3f' % (
                        i,
                        self.class_names[best_class_indices[i]],
                        best_class_probabilities[i])
                    )
            # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))

        add_overlays(frame, bounding_boxes, frame_rate, labels=labels)
        return bounding_boxes, labels


def add_overlays(frame, boxes, frame_rate, labels=None):
    if boxes is not None:
        for face in boxes:
            face_bb = face.astype(int)
            cv2.rectangle(
                frame,
                (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                (0, 255, 0), 2
            )

    if frame_rate is not None and frame_rate != 0:
        cv2.putText(
            frame, str(frame_rate) + " fps", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            thickness=2, lineType=2
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    if frame.shape[0] > 1000:
        scale = 1.0

    thickness = 1
    if labels:
        for l in labels:
            y_size = cv2.getTextSize(l['label'], font, scale, thickness=thickness)[0][1]
            top = l['top'] - 5
            if top < y_size:
                top = l['bottom'] + y_size

            cv2.putText(
                frame, l['label'], (l['left'], top),
                font, scale,
                (0, 255, 0),
                thickness=thickness, lineType=cv2.LINE_AA
            )


def get_images(image, bounding_boxes, face_crop_size=160, face_crop_margin=32, prewhiten=True):
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
            if prewhiten:
                images.append(utils.prewhiten(scaled))
            else:
                images.append(scaled)

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
