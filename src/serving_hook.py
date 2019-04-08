import base64
import json
import logging

import cv2
import numpy as np
from openvino import inference_engine as ie

from align import detect_face
import camera_openvino as ko
import openvino_detection as od
import openvino_nets as nets

LOG = logging.getLogger(__name__)
PARAMS = {
    'device': 'CPU',
    'align_model_dir': 'openvino-cpu',
    'resolutions': '26x37,37x52,52x74,145x206',
    'classifier': '',
    'threshold': [0.6, 0.7, 0.7],
    'use_tf': False,
    'use_face_detection': True,
    'face_detection_path': '',
    'tf_path': '/tf-data'
}
width = 640
height = 480
net_loaded = False
pnets = None
rnet = None
onet = None
openvino_facenet: od.OpenVINOFacenet = None


def boolean_string(s):
    if isinstance(s, bool):
        return s

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
        global openvino_facenet
        openvino_facenet = od.OpenVINOFacenet(
            kwargs.get('device'),
            PARAMS.get('face_detection_path'),
            facenet_path=None,
            classifier=[PARAMS['classifier']],
            loaded_plugin=kwargs.get('plugin'),
        )
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
        rnet_proxy = nets.RNet(plugin, net)

        LOG.info('Load ONET')

        net = ie.IENetwork.from_ir(*net_filenames(model_dir, 'onet'))
        onet_proxy = nets.ONet(plugin, net)
        onet_input_name = list(net.inputs.keys())[0]
        if isinstance(net.inputs[onet_input_name], list):
            onet_batch_size = net.inputs[onet_input_name][0]
        else:
            onet_batch_size = net.inputs[onet_input_name].shape[0]
        LOG.info('ONET_BATCH_SIZE = {}'.format(onet_batch_size))

        pnets, rnet, onet = detect_face.create_openvino_mtcnn(
            pnets_proxy, rnet_proxy, onet_proxy, onet_batch_size
        )

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
    resized = cv2.resize(image, dim, interpolation=inter)

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

    if len(image.shape) == 0:
        image = np.stack([image.tolist()])

    if len(image.shape) < 3:
        image = cv2.imdecode(np.frombuffer(image[0], np.uint8), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        # data = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
        # data = data.transpose([2, 0, 1]).reshape(1, 3, 300, 300)
        # convert to BGR
        data = image[:, :, ::-1]
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
        bounding_boxes = openvino_facenet.detect_faces(data, PARAMS['threshold'][0])
    else:
        bounding_boxes, _ = detect_face.detect_face_openvino(
            frame, pnets, rnet, onet, PARAMS['threshold']
        )
    ctx.scaled = scaled
    ctx.bounding_boxes = bounding_boxes
    ctx.frame = frame

    imgs = od.get_images(frame, bounding_boxes)

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
    box_overlays = []
    scores_out = []
    for img_idx, item_output in enumerate(facenet_output):
        if ctx.skip:
            break

        box_overlay, label = openvino_facenet.process_output(
            item_output, ctx.bounding_boxes[img_idx]
        )
        box_overlays.extend(box_overlay)
        labels.extend(label)

    table = []
    text_labels = [l['label'] for l in labels]
    for i, b in enumerate(ctx.bounding_boxes):
        x_min = int(max(0, b[0]))
        y_min = int(max(0, b[1]))
        x_max = int(min(ctx.frame.shape[1], b[2]))
        y_max = int(min(ctx.frame.shape[0], b[3]))
        cim = ctx.frame[y_min:y_max, x_min:x_max]
        # image_bytes = io.BytesIO()
        cim = cv2.cvtColor(cim, cv2.COLOR_RGB2BGR)
        image_bytes = cv2.imencode(".jpg", cim, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()

        encoded = base64.encodebytes(image_bytes).decode()
        table.append(
            {
                'type': 'text',
                'name': text_labels[i],
                'prob': float(0.0),
                'image': encoded
            }
        )
    if not ctx.skip:
        od.add_overlays(ctx.frame, box_overlays, 0, labels=labels)

    ctx.frame = cv2.cvtColor(ctx.frame, cv2.COLOR_RGB2BGR)
    image_bytes = cv2.imencode(".jpg", ctx.frame, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()

    return {
        'output': image_bytes,
        'boxes': ctx.bounding_boxes,
        'labels': np.array(text_labels, dtype=np.string_),
        'table_output': json.dumps(table),
    }
