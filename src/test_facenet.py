import argparse
import base64
import grpc
import io
import itertools
import os
from os import path
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mlboardclient.api import client
from mlboardclient import api
import numpy as np
from PIL import Image
import scipy
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

import classifier_train
import facenet

try:
    from ml_serving import predict_pb2
    from ml_serving import predict_pb2_grpc
    from ml_serving.utils import tensor_util
except ImportError:
    predict_pb2 = None
    predict_pb2_grpc = None
    tensor_util = None


MAX_LENGTH = 67108864  # 64 MB
opts = [
    ('grpc.max_send_message_length', MAX_LENGTH),
    ('grpc.max_receive_message_length', MAX_LENGTH)
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the data directory containing face dataset.'
    )
    parser.add_argument(
        '--host',
        type=str,
        help='Serving host address',
        metavar='<host>',
        default='localhost',
    )
    parser.add_argument(
        '--mlboard-url',
        type=str,
        help='ML-board address',
        metavar='<http://host:port/api/v2>',
        default=None,
    )
    parser.add_argument(
        '--use-grpc',
        action='store_true',
        help='Use grpc protocol for connection',
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Serving port number',
        metavar='<int>',
        default=9000,
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for reports',
        metavar='<dir>',
        default='.',
    )
    return parser.parse_args()


def predict_http(image_path, client, serv_addr):
    host, port = serv_addr.split(':')

    with open(image_path, 'rb') as f:
        content = f.read()

    # Do not send shaped tensor because python
    # have big compute overhead for further serialization to list and JSON
    # Send raw bytes.
    inputs = {
        "inputs": {
            "input": {"dtype": api.DT_STRING, "data": content}
        }
    }
    response = client.servings.call(
        None,
        'any',
        inputs,
        serving_address=host,
        port=port,
    )

    js = response.json()
    raw_labels = js['labels']
    labels = []
    for l in raw_labels:
        l = base64.b64decode(l.encode()).decode()
        text_label = l[l.find(' ') + 1:]
        labels.append(text_label)

    return labels


stub = None


def get_stub(addr):
    global stub
    if stub is not None:
        return stub

    channel = grpc.insecure_channel(addr, options=opts)

    stub = predict_pb2_grpc.PredictServiceStub(channel)
    return stub


def predict_grpc(data, client, serv_addr):
    tensor_proto = tensor_util.make_tensor_proto(data)
    inputs = {'input': tensor_proto}

    stub = get_stub(serv_addr)
    response = stub.Predict(predict_pb2.PredictRequest(inputs=inputs))

    raw_labels = tensor_util.make_ndarray(response.outputs['labels'])
    labels = []
    for l in raw_labels:
        l = l.decode()
        text_label = l[l.find(' ') + 1:]
        labels.append(text_label)

    return labels


def predict(image_path, client, serv_addr, use_grpc):
    if use_grpc:
        test = np.array(Image.open(image_path).convert('RGB'))
        return predict_grpc(test, client, serv_addr)
    else:
        return predict_http(image_path, client, serv_addr)


def plot_roc(classes, y_score, y_test, num_class, name):
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thr = np.zeros([num_class], dtype=np.float32)

    y = []
    for v in y_score:
        va = []
        for i in range(num_class):
            va.append(1 if v == i else 0)
        y.append(va)
    y_score = np.array(y)

    for i in range(num_class):
        y = y_test == i
        y = y.astype(int)
        fpr[i], tpr[i], th = roc_curve(y, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        score = tpr[i] + (1 - fpr[i])
        thr[i] = th[np.argmax(score)]
    y = []
    for v in y_test:
        va = []
        for i in range(num_class):
            va.append(1 if v == i else 0)
        y.append(va)
    y = np.array(y)
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = itertools.cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen']
    )
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Multi-class ROC curve for '{}'".format(name))
    plt.legend(loc="lower right")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = '<html><img src="data:image/png;base64,{}"/></html>'.format(
        base64.b64encode(buf.getvalue()).decode()
    )

    return thr, plot_data


def inside_task():
    return os.environ.get('TASK_NAME') and os.environ.get('BUILD_ID')


def update_data(data, mlboard):
    if mlboard:
        mlboard.update_task_info(data)


def upload_reports(mlboard, conf_matrix, roc):
    if not inside_task():
        return

    data = {
        '#documents.confusion_matrix.html': conf_matrix,
        '#documents.roc_curves.html': roc,
    }
    update_data(data, mlboard)


if __name__ == '__main__':
    args = parse_arguments()

    mlboard = client.Client(args.mlboard_url)
    dataset = facenet.get_dataset(args.data_dir)

    paths, labels = facenet.get_image_paths_and_labels(dataset)
    class_names = [cls.name.replace('_', ' ') for cls in dataset]

    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    print('Calculating features for images')
    nrof_images = len(paths)

    true_labels = []
    predicted_labels = []
    time_requests = 0.0
    time_all_faces = 0.0
    total_since = time.time()

    ml_serving_available = predict_pb2_grpc and predict_pb2 and tensor_util
    use_grpc = args.use_grpc and ml_serving_available
    serv_addr = '%s:%s' % (args.host, args.port)
    for i, image_path in enumerate(paths):
        true_label = labels[i]
        print('Processing {}...'.format(image_path))

        t = time.time()
        predicted = predict(image_path, mlboard, serv_addr, use_grpc)

        delta = (time.time() - t) * 1000
        time_requests += delta

        true_labels += [true_label] * len(predicted)
        predicted_labels.extend(predicted)

    indices = [class_names.index(p) for p in predicted_labels]

    print()
    print('Total images: %d' % len(paths))
    print('Total faces detected: %d' % len(predicted_labels))
    print('Total time: %.3fs' % (time.time() - total_since))
    print(
        'Average time per request: %.3fms'
        % (time_requests / float(len(paths)))
    )
    print(
        'Average time per face: %.3fms'
        % (time_requests / float(len(true_labels)))
    )
    print('Computing confusion matrix...')
    conf_matrix = classifier_train.confusion(true_labels, indices, class_names)

    print('Saving confusion matrix...')
    with open(path.join(args.output_dir, 'confusion_matrix.html'), 'w') as f:
        f.write(conf_matrix)

    print('Computing ROC curve...')
    thr, roc = plot_roc(
        class_names,
        np.array(indices),
        np.array(true_labels),
        len(class_names),
        'Face recognition'
    )
    with open(path.join(args.output_dir, 'roc_curve.html'), 'w') as f:
        f.write(roc)

    upload_reports(mlboard, conf_matrix, roc)
    print('Done.')
