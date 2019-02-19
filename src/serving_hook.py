import json
import logging
from os import path
import pickle

import cv2
import io
import numpy as np
from openvino import inference_engine as ie
from PIL import Image
import six

import camera_openvino as ko
from align import detect_face
import base64


LOG = logging.getLogger(__name__)
PARAMS = {
    'device': 'CPU',
    'align_model_dir': 'openvino-cpu',
    'resolutions': '26x37,37x52,52x74,145x206',
    'classifier': '',
    'threshold': [0.6, 0.7, 0.7],
    'use_tf': False,
    'use_face_detection': False,
    'face_detection_path': '',
    'tf_path': '/tf-data'
}
width = 640
height = 480
net_loaded = False
pnets = None
rnet = None
onet = None
model = None
face_detect = None
class_names = None


def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)

    if not isinstance(PARAMS['threshold'], list):
        PARAMS['threshold'] = [
            float(x) for x in PARAMS['threshold'].split(',')
        ]
    PARAMS['use_tf'] = boolean_string(PARAMS['use_tf'])
    LOG.info('Init with params:')
    LOG.info(json.dumps(PARAMS, indent=2))


def net_filenames(dir, net_name):
    base_name = '{}/{}'.format(dir, net_name)
    xml_name = base_name + '.xml'
    bin_name = base_name + '.bin'
    return xml_name, bin_name


class OpenVINONet(object):
    output_list = None

    def __init__(self, plugin, net):
        self.exec_net = plugin.load(net)
        if self.output_list:
            self.outputs = self.output_list
        else:
            self.outputs = list(iter(net.outputs))
        self.input = list(net.inputs.keys())[0]
        LOG.info(self.outputs)

    def __call__(self, img):
        output = self.exec_net.infer({self.input: img})
        out = [output[x] for x in self.outputs]
        if len(out) == 1:
            return out[0]
        else:
            return out


class RNet(OpenVINONet):
    output_list = [
        'rnet/conv5-2/conv5-2/MatMul',
        'rnet/prob1'
    ]


class ONet(OpenVINONet):
    output_list = [
        'onet/conv6-2/conv6-2/MatMul',
        'onet/conv6-3/conv6-3/MatMul',
        'onet/prob1'
    ]


class FaceDetect(OpenVINONet):
    output_list = ['detection_out']


def load_nets(**kwargs):
    global pnets
    global rnet
    global onet

    use_tf = PARAMS['use_tf']
    use_face = PARAMS['use_face_detection']
    if use_tf:
        model_path = PARAMS['tf_path']
        import tensorflow as tf
        sess = tf.Session()
        pnets, rnet, onet = detect_face.create_mtcnn(sess, model_path)
    elif use_face:
        LOG.info('Load FACE DETECTION')
        plugin = kwargs.get('plugin')
        model_path = PARAMS.get('face_detection_path')
        bin_path = model_path[:model_path.rfind('.')] + '.bin'

        net = ie.IENetwork.from_ir(
            path.join(model_path),
            path.join(bin_path)
        )
        global face_detect
        face_detect = FaceDetect(plugin, net)
    else:
        plugin = kwargs.get('plugin')
        model_dir = PARAMS.get('align_model_dir')

        LOG.info('Load PNET')

        pnets_proxy = []
        for r in ko.parse_resolutions(PARAMS['resolutions']):
            p = ko.PNetHandler(plugin, r[0], r[1], net_dir=model_dir)
            pnets_proxy.append(p.proxy())

        LOG.info('Load RNET')
        net = ie.IENetwork.from_ir(*net_filenames(model_dir, 'rnet'))
        rnet_proxy = RNet(plugin, net)

        LOG.info('Load ONET')

        net = ie.IENetwork.from_ir(*net_filenames(model_dir, 'onet'))
        onet_proxy = ONet(plugin, net)
        onet_input_name = list(net.inputs.keys())[0]
        if isinstance(net.inputs[onet_input_name], list):
            onet_batch_size = net.inputs[onet_input_name][0]
        else:
            onet_batch_size = net.inputs[onet_input_name].shape[0]
        LOG.info('ONET_BATCH_SIZE = {}'.format(onet_batch_size))

        pnets, rnet, onet = detect_face.create_openvino_mtcnn(
            pnets_proxy, rnet_proxy, onet_proxy, onet_batch_size
        )

    LOG.info('Load classifier')
    with open(PARAMS['classifier'], 'rb') as f:
        global model
        global class_names
        opts = {'file': f}
        if six.PY3:
            opts['encoding'] = 'latin1'
        (model, class_names) = pickle.load(**opts)
    LOG.info('Done.')


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def preprocess(inputs, ctx, **kwargs):
    global net_loaded
    if not net_loaded:
        load_nets(**kwargs)
        net_loaded = True

    image = inputs.get('input')
    image_pil = None
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    if len(image.shape) == 0:
        image = [image.tolist()]

    if isinstance(image[0], (six.string_types, bytes)):
        image = Image.open(io.BytesIO(image[0]))

        image_pil = image.convert('RGB')
        image = np.array(image_pil)

    if image.shape[2] == 4:
        # Convert RGBA -> RGB
        rgba_image = Image.fromarray(image)
        image_pil = rgba_image.convert('RGB')
        image = np.array(image_pil)

    use_tf = PARAMS['use_tf']
    use_face = PARAMS['use_face_detection']
    frame = image
    scaled = (1, 1)

    if use_tf:
        if image.shape[0] > height:
            frame = image_resize(image, height=height)
        elif image.shape[1] > width:
            frame = image_resize(image, width=width)
            scaled = (float(width) / image.shape[1], float(height) / image.shape[0])
    elif use_face:
        if image_pil is None:
            image_pil = Image.fromarray(image)

        data = image_pil.resize((300, 300), Image.ANTIALIAS)
        data = np.array(data).transpose([2, 0, 1]).reshape(1, 3, 300, 300)
        # convert to BGR
        data = data[:, ::-1, :, :]
        frame = image
    else:
        if image.shape[0] != height or image.shape[1] != width:
            frame = cv2.resize(
                image, (width, height), interpolation=cv2.INTER_AREA
            )
            scaled = (float(width) / frame.shape[1], float(height) / frame.shape[0])

    if use_tf:
        bounding_boxes, _ = detect_face.detect_face(
            frame, 20, pnets, rnet, onet, PARAMS['threshold'], 0.709
        )
    elif use_face:
        raw = face_detect(data).reshape([-1, 7])
        # 7 values:
        # class_id, label, confidence, x_min, y_min, x_max, y_max
        # Select boxes where confidence > factor
        bboxes_raw = raw[raw[:, 2] > PARAMS['threshold'][0]]
        bboxes_raw[:, 3] = bboxes_raw[:, 3] * image_pil.width
        bboxes_raw[:, 5] = bboxes_raw[:, 5] * image_pil.width
        bboxes_raw[:, 4] = bboxes_raw[:, 4] * image_pil.height
        bboxes_raw[:, 6] = bboxes_raw[:, 6] * image_pil.height

        bounding_boxes = np.zeros([len(bboxes_raw), 5])

        bounding_boxes[:, 0:4] = bboxes_raw[:, 3:7]
        bounding_boxes[:, 4] = bboxes_raw[:, 2]
    else:
        bounding_boxes, _ = detect_face.detect_face_openvino(
            frame, pnets, rnet, onet, PARAMS['threshold']
        )
    ctx.scaled = scaled
    ctx.bounding_boxes = bounding_boxes
    ctx.frame = frame

    imgs = ko.get_images(frame, bounding_boxes)

    if len(imgs) > 0:
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        ctx.skip = False
    else:
        imgs = np.random.randn(1, 3, 160, 160).astype(np.float32)
        ctx.skip = True

    model_input = list(kwargs['model_inputs'].keys())[0]
    return {model_input: imgs}


def postprocess(outputs, ctx, **kwargs):
    facenet_output = list(outputs.values())[0]
    LOG.info('output shape = {}'.format(facenet_output.shape))

    labels = []
    scores_out = []
    table_text = []
    for img_idx, item_output in enumerate(facenet_output):
        if ctx.skip:
            break

        output = item_output.reshape(1, model.shape_fit_[1])
        predictions = model.predict_proba(output)

        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[
            np.arange(len(best_class_indices)),
            best_class_indices
        ]

        for i in range(len(best_class_indices)):
            bb = ctx.bounding_boxes[img_idx].astype(int)
            text = '%.1f%% %s' % (
                best_class_probabilities[i] * 100,
                class_names[best_class_indices[i]]
            )
            table_text.append(class_names[best_class_indices[i]])
            scores_out.append(float(best_class_probabilities[i]))
            labels.append({
                'label': text,
                'left': bb[0],
                'top': bb[1] - 5
            })
            # DEBUG
            LOG.info('%2d. %s: %.3f' % (
                img_idx,
                class_names[best_class_indices[i]],
                best_class_probabilities[i])
            )

    table = []
    text_labels = [l['label'] for l in labels]
    for i,b in enumerate(ctx.bounding_boxes):
        x_min = max(0,b[0])
        y_min = max(0,b[1])
        x_max = min(ctx.frame.shape[1],b[2])
        y_max = min(ctx.frame.shape[0],b[3])
        cim = ctx.frame[y_min:y_max,x_min:x_max]
        image_bytes = io.BytesIO()
        im = Image.fromarray(cim)
        im.save(image_bytes, format='PNG')

        encoded = base64.encodebytes(image_bytes.getvalue()).decode()
        table.append(
            {
                'type': 'text',
                'name': table_text[i],
                'probability': float(scores_out[i]),
                'image': encoded
            }
        )
    if not ctx.skip:
        ko.add_overlays(ctx.frame, ctx.bounding_boxes, 0, labels=labels)

    image_bytes = io.BytesIO()

    im = Image.fromarray(ctx.frame)
    im.save(image_bytes, format='PNG')


    return {
        'output': image_bytes.getvalue(),
        'boxes': ctx.bounding_boxes,
        'labels': np.array(text_labels, dtype=np.string_)
        'table_output': json.dumps(table),
    }

