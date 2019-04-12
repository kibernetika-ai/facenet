import os
import pickle

import cv2
import numpy as np
import six
from openvino import inference_engine as ie
from sklearn import neighbors
from sklearn import svm

import bg_remove
import openvino_nets as nets
import utils


class OpenVINOFacenet(object):
    def __init__(
            self,
            device,
            face_detection_path,
            facenet_path=None,
            classifier=None,
            bg_remove_path=None,
            loaded_plugin=None,
            debug=False):

        self.use_classifiers = False
        self.debug = debug

        extensions = os.environ.get('INTEL_EXTENSIONS_PATH')
        if loaded_plugin is not None:
            plugin = loaded_plugin
        else:
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

        if facenet_path:
            print('Load FACENET')

            model_file = facenet_path
            weights_file = model_file[:model_file.rfind('.')] + '.bin'

            net = ie.IENetwork(model_file, weights_file)
            self.facenet_input = list(net.inputs.keys())[0]
            outputs = list(iter(net.outputs))
            self.facenet_output = outputs[0]
            self.face_net = plugin.load(net)

        if classifier and len(classifier) > 0:
            self.use_classifiers = bool(facenet_path)
            self.classifiers = []
            self.classifier_names = []
            self.embedding_sizes = []
            self.class_names = None
            self.class_stats = None

            for clfi, clf in enumerate(classifier):
                # Load classifier
                with open(clf, 'rb') as f:
                    print('Load CLASSIFIER %s' % clf)
                    opts = {'file': f}
                    if six.PY3:
                        opts['encoding'] = 'latin1'
                    (classifier, class_names, class_stats) = pickle.load(**opts)
                    if isinstance(classifier, svm.SVC):
                        embedding_size = classifier.shape_fit_[1]
                        clfn = "SVM classifier"
                        self.classifier_names.append("SVM")
                    elif isinstance(classifier, neighbors.KNeighborsClassifier):
                        embedding_size = classifier._fit_X.shape[1]
                        clfn = "kNN (neighbors %d) classifier" % classifier.n_neighbors
                        # self.classifier_names.append("kNN(%2d)" % classifier.n_neighbors)
                        self.classifier_names.append("kNN")
                    else:
                        # try embedding_size = 512
                        embedding_size = 512
                        clfn = type(classifier)
                        self.classifier_names.append("%d" % clfi)
                    print('Loaded %s, embedding size: %d' % (clfn, embedding_size))
                    if self.class_names is None:
                        self.class_names = class_names
                    elif class_names != self.class_names:
                        raise RuntimeError("Different class names in classifiers")
                    if self.class_stats is None:
                        self.class_stats = class_stats
                    elif class_stats != self.class_stats:
                        raise RuntimeError("Different class stats in classifiers")
                    self.embedding_sizes.append(embedding_size)
                    self.classifiers.append(classifier)

        self.bg_remove = bg_remove.get_driver(bg_remove_path)

    def detect_faces(self, frame, threshold=0.5):
        if self.bg_remove is not None:
            bounding_boxes_frame = self.bg_remove.apply_mask(frame)
        else:
            bounding_boxes_frame = frame

        return openvino_detect(self.face_detect, bounding_boxes_frame, threshold)

    def inference_facenet(self, img):
        output = self.face_net.infer(inputs={self.facenet_input: img})

        return output[self.facenet_output]

    def process_output(self, output, bbox):
        detected_indices = []
        label_strings = []
        probs = []
        prob_detected = True

        for clfi, clf in enumerate(self.classifiers):

            try:
                output = output.reshape(1, self.embedding_sizes[clfi])
                predictions = clf.predict_proba(output)
            except ValueError as e:
                # Can not reshape
                print("ERROR: Output from graph doesn't consistent with classifier model: %s" % e)
                continue

            best_class_indices = np.argmax(predictions, axis=1)

            if isinstance(clf, neighbors.KNeighborsClassifier):

                def process_index(idx):
                    cnt = self.class_stats[best_class_indices[idx]]['embeddings']
                    (closest_distances, neighbors_indices) = clf.kneighbors(output, n_neighbors=cnt)
                    eval_values = closest_distances[:, 0]
                    first_cnt = 0
                    for i in neighbors_indices[0]:
                        if clf._y[i] != best_class_indices[idx]:
                            break
                        first_cnt += 1
                    # probability:
                    # first matched embeddings
                    # less than 25% is 0%, more than 75% is 100%
                    # multiplied by distance coefficient:
                    # 0.5 and less is 100%, 0.83 and more is 0%
                    prob = max(0, min(1, 2 * first_cnt / cnt - .5)) * max(0, min(1, 2.5 - eval_values[idx] * 3))
                    label_debug = '%.3f %d/%d' % (
                        eval_values[idx],
                        first_cnt, cnt,
                    )
                    return prob, label_debug

            elif isinstance(clf, svm.SVC):

                def process_index(idx):
                    eval_values = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    label_debug = '%.1f%%' % (eval_values[idx] * 100)
                    return max(0, min(1, eval_values[idx] * 10)), label_debug

            else:

                print("ERROR: Unsupported model type: %s" % type(clf))
                continue

            for i in range(len(best_class_indices)):
                detected_indices.append(best_class_indices[i])
                overlay_label = self.class_names[best_class_indices[i]]
                prob, label_debug = process_index(i)
                probs.append(prob)
                if prob <= 0:
                    prob_detected = False
                label_debug_info = '%s: %.1f%% %s (%s)' % (
                self.classifier_names[clfi], prob * 100, overlay_label, label_debug)
                if self.debug:
                    label_strings.append(label_debug_info)
                elif len(label_strings) == 0:
                    label_strings.append(overlay_label)
                print(label_debug_info)

        # detected if all classes are the same, and all probs are more than 0
        detected = len(set(detected_indices)) == 1 and prob_detected
        mean_prob = sum(probs) / len(probs) if detected else 0

        if self.debug:
            if detected:
                label_strings.append("Summary: %.1f%% %s" % (mean_prob * 100, overlay_label))
            else:
                label_strings.append("Summary: not detected")

        thin = not detected
        color = (0, 0, 255) if thin else (0, 255, 0)

        bb = bbox.astype(int)
        bounding_boxes_overlay = {
            'bb': bb,
            'thin': thin,
            'color': color,
        }

        overlay_label_str = ""
        if self.debug:
            if len(label_strings) > 0:
                overlay_label_str = "\n".join(label_strings)
        elif detected:
            overlay_label_str = label_strings[0]

        overlay_label = None
        if overlay_label_str != "":
            overlay_label = {
                'label': overlay_label_str,
                'left': bb[0],
                'top': bb[1],
                'right': bb[2],
                'bottom': bb[3],
                'color': color,
            }

        return bounding_boxes_overlay, overlay_label, mean_prob

    def process_frame(self, frame, threshold=0.5, frame_rate=None, overlays=True):
        bounding_boxes_detected = self.detect_faces(frame, threshold)

        bounding_boxes_overlays = []
        labels = []
        if self.use_classifiers:

            imgs = get_images(frame, bounding_boxes_detected)

            for img_idx, img in enumerate(imgs):

                label_strings = []

                # Infer
                # t = time.time()
                # Convert BGR to RGB
                img = img[:, :, ::-1]
                img = img.transpose([2, 0, 1]).reshape([1, 3, 160, 160])
                output = self.inference_facenet(img)
                # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))
                # output = output[facenet_output]

                face_overlay, face_label, _ = self.process_output(output, bounding_boxes_detected[img_idx])
                bounding_boxes_overlays.append(face_overlay)
                if face_label:
                    labels.append(face_label)

        # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))
        if overlays:
            self.add_overlays(frame, bounding_boxes_overlays, labels, frame_rate)
        return bounding_boxes_overlays, labels

    @staticmethod
    def add_overlays(frame, boxes, labels=None, frame_rate=None, align_to_right=True):
        add_overlays(
            frame, boxes,
            frame_rate=frame_rate,
            labels=labels,
            align_to_right=align_to_right
        )


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
