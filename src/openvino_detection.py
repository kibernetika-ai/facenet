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

from ml_serving.drivers import driver


class OpenVINOFacenet(object):
    def __init__(self, device, face_detection_path, facenet_path=None, classifier=None, bg_remove_path=None, debug=False):

        self.use_classifiers = False
        self.bg_remove = False
        self.debug = debug

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

        if facenet_path and classifier and len(classifier) > 0:
            self.use_classifiers = True
            print('Load FACENET')

            model_file = facenet_path
            weights_file = model_file[:model_file.rfind('.')] + '.bin'

            net = ie.IENetwork(model_file, weights_file)
            self.facenet_input = list(net.inputs.keys())[0]
            outputs = list(iter(net.outputs))
            self.facenet_output = outputs[0]
            self.face_net = plugin.load(net)

            self.classifiers = []
            self.classifier_names = []
            self.embedding_sizes = []
            self.class_names = None

            for clfi, clf in enumerate(classifier):
                # Load classifier
                with open(clf, 'rb') as f:
                    print('Load CLASSIFIER %s' % clf)
                    opts = {'file': f}
                    if six.PY3:
                        opts['encoding'] = 'latin1'
                    (classifier, class_names) = pickle.load(**opts)
                    if isinstance(classifier, svm.SVC):
                        embedding_size = classifier.shape_fit_[1]
                        clfn = "SVM classifier"
                        self.classifier_names.append("SVM")
                    elif isinstance(classifier, neighbors.KNeighborsClassifier):
                        embedding_size = classifier._fit_X.shape[1]
                        clfn = "kNN (neighbors %d) classifier" % classifier.n_neighbors
                        self.classifier_names.append("kNN(%2d)" % classifier.n_neighbors)
                    else:
                        # try embedding_size = 512
                        embedding_size = 512
                        clfn = type(classifier)
                        self.classifier_names.append("%d" % clfi)
                    print('Loaded %s, embedding size: %d' % (clfn, embedding_size))
                    if self.class_names is None:
                        self.class_names = class_names
                    else:
                        if class_names != self.class_names:
                            print("Different class names in classificators")
                            print(class_names)
                            print(self.class_names)
                            exit(1)
                    self.embedding_sizes.append(embedding_size)
                    self.classifiers.append(classifier)

        if bg_remove_path:
            self.bg_remove = True
            print('Load bg_remove model')
            drv = driver.load_driver('tensorflow')
            self.bg_remove_drv = drv()
            self.bg_remove_drv.load_model(bg_remove_path)

    def process_frame(self, frame, threshold=0.5, frame_rate=None):

        if self.bg_remove and self.bg_remove_drv:
            input = cv2.resize(frame[:, :, ::-1].astype(np.float32), (160, 160)) / 255.0
            outputs = self.bg_remove_drv.predict({'image': np.expand_dims(input, 0)})
            mask = cv2.resize(outputs['output'][0], (frame.shape[1], frame.shape[0]))
            bounding_boxes_frame = frame * np.expand_dims(mask, 2)
        else:
            bounding_boxes_frame = frame

        bounding_boxes_detected = openvino_detect(self.face_detect, bounding_boxes_frame, threshold)

        bounding_boxes = []
        bounding_boxes_overlays = []
        labels = []
        if self.use_classifiers:

            imgs = get_images(frame, bounding_boxes_detected)

            for img_idx, img in enumerate(imgs):

                label_strings = []
                color = (0, 255, 0)

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

                for clfi, clf in enumerate(self.classifiers):

                    try:
                        output = output.reshape(1, self.embedding_sizes[clfi])
                        predictions = clf.predict_proba(output)
                    except ValueError as e:
                        # Can not reshape
                        print(
                            "ERROR: Output from graph doesn't consistent"
                            " with classifier model: %s" % e
                        )
                        continue


                    best_class_indices = np.argmax(predictions, axis=1)

                    if isinstance(clf, neighbors.KNeighborsClassifier):

                        def get_label(idx):

                            (closest_distances, neighbors_indices) = clf.kneighbors(output, n_neighbors=30)
                            eval_values = closest_distances[:, 0]
                            first_cnt = 1
                            for i in neighbors_indices[0]:
                                if clf._y[i] != best_class_indices[idx]:
                                    break
                                first_cnt += 1
                            cnt = len(clf._y[clf._y == best_class_indices[idx]])
                            return '%s (%.3f %d/%d %.1f%%)' % (
                                self.class_names[best_class_indices[idx]],
                                eval_values[idx],
                                first_cnt , cnt,
                                first_cnt / cnt * 100,
                            )

                        def is_skipped(value):
                            if self.debug:
                                return False
                            return value > 0.9

                        def is_recognized(value):
                            return value <= 0.8

                    else:

                        def get_label(idx):
                            eval_values = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            return '%s (%.1f%%)' % (
                                self.class_names[best_class_indices[idx]],
                                eval_values[idx] * 100,
                            )

                        def is_skipped(value):
                            return value == 0

                        def is_recognized(value):
                            return value >= 30


                    for i in range(len(best_class_indices)):

                        label = get_label(i)
                        if len(self.classifier_names) > 1:
                            label = '%s: %s' % (self.classifier_names[clfi], label)

                        label_strings.append(label)
                        print(label)

                bb = bounding_boxes_detected[img_idx].astype(int)
                bounding_boxes.append(bb)
                bounding_boxes_overlays.append({
                    'bb': bb,
                    'thin': False,
                    'color': color,
                })
                if len(label_strings) > 0:
                    labels.append({
                        'label': "\n".join(label_strings),
                        'left': bb[0],
                        'top': bb[1],
                        'right': bb[2],
                        'bottom': bb[3],
                        'color': color,
                    })



        # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))

        add_overlays(frame, bounding_boxes_overlays, frame_rate, labels=labels)
        return bounding_boxes, labels


def add_overlays(frame, boxes, frame_rate=None, labels=None, align_to_right=True):
    if boxes is not None:
        for face in boxes:
            face_bb = face['bb'].astype(int)
            cv2.rectangle(
                frame,
                (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                face['color'], 1 if face['thin'] else 2,
            )

    frame_avg = (frame.shape[1] + frame.shape[0]) / 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = frame_avg / 1300
    font_thickness = 2 if frame_avg > 1000 else 1
    font_inner_padding_w, font_inner_padding_h = 5, 5

    if frame_rate is not None and frame_rate != 0:
        fps_txt = "%d fps" % frame_rate
        _, flh = cv2.getTextSize(fps_txt, font, font_size, thickness=font_thickness)[0]
        cv2.putText(
            frame, fps_txt,
            (font_inner_padding_w, font_inner_padding_h + flh),
            font, font_size, (0, 255, 0),
            thickness=font_thickness, lineType=2
        )

    if labels:
        for l in labels:
            strs = l['label'].split('\n')
            str_w, str_h = 0, 0
            widths = []
            for i, line in enumerate(strs):
                lw, lh = cv2.getTextSize(line, font, font_size, thickness=font_thickness)[0]
                str_w = max(str_w, lw)
                str_h = max(str_h, lh)
                widths.append(lw)
            str_h = int(str_h * 1.6) # line height

            to_right = l['left'] + str_w > frame.shape[1] - font_inner_padding_w

            top = l['top'] - int((len(strs) - 0.5) * str_h)
            if top < str_h + font_inner_padding_h:
                top = min(l['bottom'] + int(str_h * 1.2), frame.shape[0] - str_h * len(strs) + font_inner_padding_h)

            for i, line in enumerate(strs):
                if align_to_right:
                    # all align to right box border
                    left = (l['right'] - widths[i] - font_inner_padding_w) if to_right else l['left'] + font_inner_padding_w
                else:
                    # move left each string if it's ending not places on the frame
                    left = frame.shape[1] - widths[i] - font_inner_padding_w \
                        if l['left'] + widths[i] > frame.shape[1] - font_inner_padding_w \
                        else l['left'] + font_inner_padding_w

                cv2.putText(
                    frame, line,
                    (
                        left,
                        int(top + i * str_h),
                    ),
                    font,
                    font_size,
                    l['color'],
                    thickness=font_thickness, lineType=cv2.LINE_AA
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
