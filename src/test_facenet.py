import argparse
import grpc
import time

import imageio
from ml_serving import predict_pb2
from ml_serving import predict_pb2_grpc
from ml_serving.utils import tensor_util
import numpy as np

import classifier_train
import facenet


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
        '--port',
        type=int,
        help='Serving port number',
        metavar='<int>',
        default=9000,
    )
    return parser.parse_args()


def predict(image_path, stub):
    test = imageio.imread(image_path)
    tensor_proto = tensor_util.make_tensor_proto(test)
    inputs = {'input': tensor_proto}

    response = stub.Predict(predict_pb2.PredictRequest(inputs=inputs))

    raw_labels = tensor_util.make_ndarray(response.outputs['labels'])
    labels = []
    for l in raw_labels:
        l = l.decode()
        text_label = l[l.find(' ')+1:]
        labels.append(text_label)

    return labels


if __name__ == '__main__':
    args = parse_arguments()
    dataset = facenet.get_dataset(args.data_dir)

    paths, labels = facenet.get_image_paths_and_labels(dataset)
    class_names = [cls.name.replace('_', ' ') for cls in dataset]

    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    print('Calculating features for images')
    nrof_images = len(paths)

    server = '%s:%s' % (args.host, args.port)
    channel = grpc.insecure_channel(server)

    stub = predict_pb2_grpc.PredictServiceStub(channel)

    true_labels = []
    predicted_labels = []
    time_requests = 0.0
    time_all_faces = 0.0

    for i, path in enumerate(paths):
        true_label = labels[i]
        print('Processing {}...'.format(path))

        t = time.time()
        predicted = predict(path, stub)

        delta = (time.time() - t) * 1000
        time_requests += delta

        true_labels += [true_label] * len(predicted)
        predicted_labels.extend(predicted)

    indices = [class_names.index(p) for p in predicted_labels]

    print()
    print('Average time per request: %.3fms' % (time_requests / float(len(paths))))
    print('Average time per face: %.3fms' % (time_requests / float(len(true_labels))))
    print('Computing confusion matrix...')
    rpt = classifier_train.confusion(true_labels, indices, class_names)

    print('Saving confusion matrix...')
    with open('confusion_matrix.html', 'w') as f:
        f.write(rpt)

    print('Done.')
