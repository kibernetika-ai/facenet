"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import math
import os
from os import path
import pickle
import sys
import shutil
import time

import numpy as np
from sklearn import svm
from sklearn import neighbors

import facenet
from ml_serving.drivers import driver
from tensorflow import logging

try:
    from mlboardclient.api import client
except ImportError:
    client = None


LOG = logging


def print_fun(s):
    print(s)
    sys.stdout.flush()


def update_data(data, use_mlboard, mlboard):
    if use_mlboard and mlboard:
        mlboard.update_task_info(data)


def catalog_ref(name, ctype, version):
    return '#/{}/catalog/{}/{}/versions/{}'. \
        format(os.environ.get('WORKSPACE_NAME'), ctype, name, version)


def upload_model(use_mlboard, mlboard, classifier_path, model, version):
    if not use_mlboard or not mlboard:
        return

    print_fun('Uploading model...')
    dirname = '/tmp/classifier'
    os.makedirs(dirname)
    shutil.copy(classifier_path, path.join(dirname, path.basename(classifier_path)))
    # shutil.shutil.copy()
    mlboard.model_upload(model, version, dirname)

    shutil.rmtree(dirname)
    update_data({'model_reference': catalog_ref(model, 'mlmodel', version)}, use_mlboard, mlboard)
    print_fun("New model uploaded as '%s', version '%s'." % (model, version))


def confusion(y_test, y_score, labels, use_mlboard):
    from sklearn.metrics import confusion_matrix
    import itertools
    import matplotlib.pyplot as plt
    import io
    import base64
    def _plot_confusion_matrix(cm, classes, use_mlboard,
                               normalize=False,
                               title='Confusion matrix',
                               cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print_fun("Normalized confusion matrix")
        else:
            print_fun('Confusion matrix, without normalization')

        print_fun(cm)

        if not use_mlboard:
            return ''

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout(pad=1.5)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return '<html><img src="data:image/png;base64,{}"/></html>'.format(base64.b64encode(buf.getvalue()).decode())

    cm = confusion_matrix(y_test, y_score)
    return _plot_confusion_matrix(cm, labels, use_mlboard)


def main(args):
    use_mlboard = False
    mlboard = None
    if client:
        mlboard = client.Client()
        try:
            mlboard.apps.get()
        except Exception:
            mlboard = None
            print_fun('Do not use mlboard.')
        else:
            print_fun('Use mlboard parameters logging.')
            use_mlboard = True

    if args.use_split_dataset:
        dataset_tmp = facenet.get_dataset(args.data_dir)
        train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                            args.nrof_train_images_per_class)
        if args.mode == 'TRAIN':
            dataset = train_set
        elif args.mode == 'CLASSIFY':
            dataset = test_set
    else:
        dataset = facenet.get_dataset(args.data_dir)

    update_data({'mode': args.mode}, use_mlboard, mlboard)

    # Check that there are at least one training image per class
    for cls in dataset:
        if len(cls.image_paths) == 0:
            print_fun('WARNING: %s: There are no aligned images in this class.' % cls)

    paths, labels = facenet.get_image_paths_and_labels(dataset)

    print_fun('Number of classes: %d' % len(dataset))
    print_fun('Number of images: %d' % len(paths))
    data = {
        'num_classes': len(dataset),
        'num_images': len(paths),
        'model_path': args.model,
        'image_size': args.image_size,
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
    }
    update_data(data, use_mlboard, mlboard)

    # Load the model
    print_fun('Loading feature extraction model')

    # Load driver
    drv = driver.load_driver(args.driver)
    # Instantinate driver
    serving = drv()
    serving.load_model(
        args.model,
        inputs='input:0,phase_train:0',
        outputs='embeddings:0',
        device=args.device,
        flexible_batch_size=True,
    )

    # Run forward pass to calculate embeddings
    print_fun('Calculating features for images')

    total_time = 0.

    nrof_images = len(paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
    embeddings_size = nrof_images
    if args.noise:
        embeddings_size += nrof_images * args.noise_count

    if args.rotate:
        embeddings_size += nrof_images * args.rotate_count

    emb_array = np.zeros((embeddings_size, 512))
    for i in range(nrof_batches_per_epoch):
        start_index = i * args.batch_size
        end_index = min((i + 1) * args.batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]

        if args.noise:
            start_index += i * args.noise_count
            end_index += i * args.noise_count

        if args.rotate:
            start_index += i * args.rotate_count
            end_index += i * args.rotate_count

        for j in range(end_index - start_index):
            print_fun('Batch {} <-> {}'.format(paths_batch[j], labels[start_index + j]))
        images = facenet.load_data(paths_batch, False, False, args.image_size)

        images_size = len(images)
        if args.noise:
            for k in range(images_size):
                img = images[k]
                for i in range(args.noise_count):
                    print_fun(
                        'Applying noise to image {}, #{}'.format(
                            paths_batch[k], i + 1
                        )
                    )
                    noised = facenet.random_noise(img)

                    # Expand labels
                    labels.insert(start_index+k, labels[start_index+k])
                    # Add image to list
                    images = np.concatenate((images, noised.reshape(1, *noised.shape)))
                    end_index += 1

        if args.rotate:
            for k in range(images_size):
                img = images[k]
                for i in range(args.rotate_count):
                    print_fun(
                        'Applying rotate to image {}, #{}'.format(
                            paths_batch[k], i + 1
                        )
                    )
                    rotated = facenet.rotate(img)
                    # Expand labels
                    labels.insert(start_index+k, labels[start_index+k])
                    # Add image to list
                    images = np.concatenate((images, rotated.reshape(1, *rotated.shape)))
                    end_index += 1

        if serving.driver_name == 'tensorflow':
            feed_dict = {'input:0': images, 'phase_train:0': False}
        elif serving.driver_name == 'openvino':
            input_name = list(serving.inputs.keys())[0]

            # Transpose image for channel first format
            images = images.transpose([0, 3, 1, 2])
            feed_dict = {input_name: images}
        else:
            raise RuntimeError('Driver %s currently not supported' % serving.driver_name)

        t = time.time()
        outputs = serving.predict(feed_dict)
        total_time += time.time() - t
        emb_array[start_index:end_index, :] = list(outputs.values())[0]

    classifier_filename_exp = os.path.expanduser(args.classifier)
    average_time = total_time / embeddings_size * 1000
    print_fun('Average time: %.3fms' % average_time)

    if args.mode == 'TRAIN':
        # Train classifier
        model = None
        print_fun('Classifier algorithm %s' % args.algorithm)
        update_data({'classifier_algorithm': args.algorithm}, use_mlboard, mlboard)
        if args.algorithm == 'kNN':
            # n_neighbors = int(round(np.sqrt(len(emb_array))))
            model = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
        else:
            model = svm.SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        print_fun('Classes:')
        print_fun(class_names)

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile, protocol=2)
        print_fun('Saved classifier model to file "%s"' % classifier_filename_exp)
        update_data({'average_time': '%.3fms' % average_time}, use_mlboard, mlboard)

    elif args.mode == 'CLASSIFY':
        # Classify images
        print_fun('Testing classifier')
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print_fun('Loaded classifier model from file "%s"' % classifier_filename_exp)

        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print_fun('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        accuracy = np.mean(np.equal(best_class_indices, labels))

        if mlboard and use_mlboard:
            rpt = confusion(labels, best_class_indices, class_names)
            data = {
                'accuracy': accuracy,
                '#documents.confusion_matrix.html': rpt,
                'average_time': '%.3fms' % average_time
            }

            update_data(data, use_mlboard, mlboard)

        print_fun('Accuracy: %.3f' % accuracy)
        if args.upload_model and accuracy >= args.upload_threshold:
            timestamp = datetime.datetime.now().strftime('%s')
            model_name = 'facenet-classifier'

            if args.device == 'MYRIAD':
                model_name = model_name + "-movidius"

            version = '1.0.0-%s-%s' % (args.driver, timestamp)

            print_fun('Uploading model as %s:%s' % (model_name, version))
            upload_model(
                use_mlboard,
                mlboard,
                classifier_filename_exp,
                model_name,
                version
            )


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'mode',
        type=str,
        choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' +
             'model should be used for classification',
        default='CLASSIFY'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the data directory containing aligned LFW face patches.'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to .xml openVINO IR file',
        required=True,
    )
    parser.add_argument(
        '--classifier',
        help='Classifier model file name as a pickle (.pkl) file. ' +
             'For training this is the output and for classification this is an input.',
        required=True,
    )
    parser.add_argument(
        '--algorithm',
        help='Classifier algorithm.',
        default="SVM",
        choices=["SVM", "kNN"],
    )
    parser.add_argument(
        '--use_split_dataset',
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
             'Otherwise a separate test set can be specified using the test_data_dir option.',
        action='store_true'
    )
    parser.add_argument(
        '--device',
        help='Device for openVINO.',
        default="CPU",
        choices=["CPU", "MYRIAD"]
    )
    parser.add_argument(
        '--driver',
        help='Driver for inference.',
        default="openvino",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Number of images to process in a batch.',
        default=1
    )
    parser.add_argument(
        '--noise',
        action='store_true',
        help='Add random noise to images.',
    )
    parser.add_argument(
        '--noise-count',
        type=int,
        default=1,
        help='Noise count for each image.',
    )
    parser.add_argument(
        '--rotate',
        action='store_true',
        help='Add random rotate to images.',
    )
    parser.add_argument(
        '--rotate-count',
        type=int,
        default=1,
        help='Rotate count for each image.',
    )
    parser.add_argument(
        '--image_size',
        type=int,
        help='Image size (height, width) in pixels.',
        default=160
    )
    parser.add_argument(
        '--min_nrof_images_per_class',
        type=int,
        help='Only include classes with at least this number of images in the dataset',
        default=20
    )
    parser.add_argument(
        '--nrof_train_images_per_class',
        type=int,
        help='Use this number of images from each class for training and the rest for testing',
        default=10
    )
    parser.add_argument(
        '--upload-threshold',
        type=float,
        default=0.9,
        help='Threshold for uploading model',
    )
    parser.add_argument(
        '--upload-model',
        action='store_true',
        default=False,
    )

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
