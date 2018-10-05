import logging
import pickle

import cv2
import io
import numpy as np
from openvino import inference_engine as ie
from PIL import Image
import six

import camera_openvino as ko
from align import detect_face


LOG = logging.getLogger(__name__)
PARAMS = {
    'device': 'CPU',
    'align_model_dir': 'openvino-cpu',
    'resolutions': '26x37,37x52,52x74,145x206',
    'classifier': '',
    'threshold': [0.6, 0.7, 0.7],
    'use_tf': False,
    'tf_path': '/tf-data'
}
width = 640
height = 480
net_loaded = False
pnets = None
rnet = None
onet = None
model = None
class_names = None


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)

    if not isinstance(PARAMS['threshold'], list):
        PARAMS['threshold'] = [
            float(x) for x in PARAMS['threshold'].split(',')
        ]
    PARAMS['use_tf'] = bool(PARAMS['use_tf'])


def net_filenames(dir, net_name):
    base_name = '{}/{}'.format(dir, net_name)
    xml_name = base_name + '.xml'
    bin_name = base_name + '.bin'
    return xml_name, bin_name


class OpenVINONet(object):
    def __init__(self, plugin, net):
        self.exec_net = plugin.load(net)
        self.outputs = net.outputs
        self.input = list(net.inputs.keys())[0]

    def __call__(self, img):
        output = self.exec_net.infer({self.input: img})
        out = [output[x] for x in self.outputs]
        if len(out) == 1:
            return out[0]
        else:
            return out


def load_nets(**kwargs):
    global pnets
    global rnet
    global onet

    use_tf = PARAMS['use_tf']
    if use_tf:
        model_path = PARAMS['tf_path']
        import tensorflow as tf
        sess = tf.Session()
        pnets, rnet, onet = detect_face.create_mtcnn(sess, model_path)
    else:
        plugin = kwargs.get('plugin')
        model_dir = PARAMS.get('align_model_dir')

        LOG.info('Load PNET')

        pnets_proxy = []
        for r in ko.parse_resolutions(PARAMS['resolutions']):
            p = ko.PNetHandler(plugin, r[0], r[1])
            pnets_proxy.append(p.proxy())

        LOG.info('Load RNET')
        net = ie.IENetwork.from_ir(*net_filenames(model_dir, 'rnet'))
        rnet_proxy = OpenVINONet(plugin, net)

        LOG.info('Load ONET')

        net = ie.IENetwork.from_ir(*net_filenames(model_dir, 'onet'))
        onet_proxy = OpenVINONet(plugin, net)
        onet_input_name = list(net.inputs.keys())[0]
        onet_batch_size = net.inputs[onet_input_name][0]
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
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    if isinstance(image[0], (six.string_types, bytes)):
        image = Image.open(io.BytesIO(image[0]))

        rgba_image = Image.fromarray(image)
        image = rgba_image.convert('RGB')
        image = np.array(image)

    if image.shape[2] == 4:
        # Convert RGBA -> RGB
        rgba_image = Image.fromarray(image)
        image = rgba_image.convert('RGB')
        image = np.array(image)

    use_tf = PARAMS['use_tf']
    frame = image
    scaled = (1, 1)

    if use_tf:
        if image.shape[0] > height:
            frame = image_resize(image, height=height)
        elif image.shape[1] > width:
            frame = image_resize(image, width=width)
            scaled = (float(width) / image.shape[1], float(height) / image.shape[0])
    else:
        if image.shape[0] > height or image.shape[1] > width:
            frame = cv2.resize(
                image, (width, height), interpolation=cv2.INTER_AREA
            )
            scaled = (float(width) / frame.shape[1], float(height) / frame.shape[0])

    if use_tf:
        bounding_boxes, _ = detect_face.detect_face(
            frame, 20, pnets, rnet, onet, PARAMS['threshold'], 0.709
        )
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
            labels.append({
                'label': text,
                'left': bb[0],
                'top': bb[1] - 5
            })
            # DEBUG
            LOG.info('%4d  %s: %.3f' % (
                i,
                class_names[best_class_indices[i]],
                best_class_probabilities[i])
            )

    if not ctx.skip:
        ko.add_overlays(ctx.frame, ctx.bounding_boxes, 0, labels=labels)

    image_bytes = io.BytesIO()

    im = Image.fromarray(ctx.frame)
    im.save(image_bytes, format='PNG')

    return {
        'output': image_bytes.getvalue(),
        'boxes': ctx.bounding_boxes,
        # 'labels': np.array(labels)
    }
