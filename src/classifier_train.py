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
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time

import numpy as np
from ml_serving.drivers import driver
from sklearn import neighbors
from sklearn import svm
from tensorflow import logging

import facenet

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


def upload_model(use_mlboard, mlboard, classifiers_path, model, version):
    if not use_mlboard or not mlboard:
        print_fun("Skipped: no mlboard detected")
        return

    print_fun('Uploading model...')
    # dirname = '/tmp/classifier'
    # os.makedirs(dirname)
    # shutil.copy(classifiers_path, path.join(dirname, path.basename(classifiers_path)))
    # shutil.shutil.copy()
    mlboard.model_upload(model, version, classifiers_path)

    # shutil.rmtree(dirname)
    update_data({'model_reference': catalog_ref(model, 'mlmodel', version)}, use_mlboard, mlboard)
    print_fun("New model uploaded as '%s', version '%s'." % (model, version))


def confusion(y_test, y_score, labels, draw):
    from sklearn.metrics import confusion_matrix
    import itertools
    import matplotlib.pyplot as plt
    import io
    import base64
    def _plot_confusion_matrix(cm, classes, draw,
                               normalize=False,
                               title='Confusion matrix',
                               cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print_fun("Normalized confusion matrix")
        else:
            print_fun('Confusion matrix, without normalization')

        print_fun(cm)

        if not draw:
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
    return _plot_confusion_matrix(cm, labels, draw)


def main(args):
    algorithms = ["kNN", "SVM"]

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

    # Load and instantinate driver
    drv = driver.load_driver(args.driver)
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

    noise_count = max(0, args.noise_count) if args.noise else 0
    emb_args = {
        'model': args.model,
        'use_split_dataset': args.use_split_dataset,
        'noise': noise_count > 0,
        'noise_count': noise_count,
        'flip': args.flip,
        'image_size': args.image_size,
        'min_nrof_images_per_class': args.min_nrof_images_per_class,
        'nrof_train_images_per_class': args.nrof_train_images_per_class,
    }

    stored_embeddings = {}
    if args.mode == 'TRAIN':
        embeddings_filename = os.path.join(
            args.data_dir,
            "embeddings-%s.pkl" % hashlib.md5(json.dumps(emb_args, sort_keys=True).encode()).hexdigest(),
        )
        if os.path.isfile(embeddings_filename):
            print_fun("Found stored embeddings data, loading...")
            with open(embeddings_filename, 'rb') as embeddings_file:
                stored_embeddings = pickle.load(embeddings_file)

    total_time = 0.

    nrof_images = len(paths)

    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
    epp = embeddings_per_path(noise_count, args.flip)
    embeddings_size = nrof_images * epp

    emb_array = np.zeros((embeddings_size, 512))
    fit_labels = []

    emb_index = 0
    for i in range(nrof_batches_per_epoch):
        start_index = i * args.batch_size
        end_index = min((i + 1) * args.batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        labels_batch = labels[start_index:end_index]

        # has_not_stored_embeddings = False
        paths_batch_load, labels_batch_load = [], []

        for j in range(end_index - start_index):
            # print_fun(os.path.split(paths_batch[j]))
            cls_name = dataset[labels_batch[j]].name
            cached = True
            if cls_name not in stored_embeddings or paths_batch[j] not in stored_embeddings[cls_name]:
                # has_not_stored_embeddings = True
                cached = False
                paths_batch_load.append(paths_batch[j])
                labels_batch_load.append(labels_batch[j])
            else:
                embeddings = stored_embeddings[cls_name][paths_batch[j]]
                emb_array[emb_index:emb_index + len(embeddings), :] = stored_embeddings[cls_name][paths_batch[j]]
                fit_labels.extend([labels_batch[j]] * len(embeddings))
                emb_index += len(embeddings)

            print_fun('Batch {} <-> {} {} {}'.format(
                paths_batch[j], labels_batch[j], cls_name, "cached" if cached else "",
            ))

        if len(paths_batch_load) == 0:
            continue

        images = load_data(paths_batch_load, labels_batch_load, args.image_size, noise_count, args.flip)

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

        emb_outputs = list(outputs.values())[0]

        if args.mode == "TRAIN":
            for n, e in enumerate(emb_outputs):
                cls_name = dataset[labels_batch_load[n]].name
                if cls_name not in stored_embeddings:
                    stored_embeddings[cls_name] = {}
                path = paths_batch_load[n]
                if path not in stored_embeddings[cls_name]:
                    stored_embeddings[cls_name][path] = []
                stored_embeddings[cls_name][path].append(e)

        emb_array[emb_index:emb_index + len(images), :] = emb_outputs
        fit_labels.extend(labels_batch_load)

        emb_index += len(images)

    # average_time = total_time / embeddings_size * 1000
    # print_fun('Average time: %.3fms' % average_time)

    classifiers_path = os.path.expanduser(args.classifiers_path)

    if args.mode == 'TRAIN':

        # Save embeddings
        with open(embeddings_filename, 'wb') as embeddings_file:
            pickle.dump(stored_embeddings, embeddings_file, protocol=2)

        # Clear (or create) classifiers directory
        try:
            shutil.rmtree(classifiers_path, ignore_errors=True)
        except:
            pass
        os.makedirs(classifiers_path)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        print_fun('Classes:')
        print_fun(class_names)

        # Train classifiers
        for algorithm in algorithms:
            if args.only_algorithm is not None and algorithm != args.only_algorithm:
                continue

            print_fun('Classifier algorithm %s' % algorithm)
            # update_data({'classifier_algorithm': args.algorithm}, use_mlboard, mlboard)
            if algorithm == 'SVM':
                model = svm.SVC(kernel='linear', probability=True)
            elif algorithm == 'kNN':
                # n_neighbors = int(round(np.sqrt(len(emb_array))))
                model = neighbors.KNeighborsClassifier(n_neighbors=args.knn_neighbors, weights='distance')
            else:
                raise RuntimeError("Classifier algorithm %s not supported" % algorithm)

            model.fit(emb_array, fit_labels)

            # Saving classifier model
            classifier_filename = get_classifier_path(classifiers_path, algorithm)
            with open(classifier_filename, 'wb') as outfile:
                pickle.dump((model, class_names), outfile, protocol=2)
            print_fun('Saved classifier model to file "%s"' % classifier_filename)
            # update_data({'average_time_%s': '%.3fms' % average_time}, use_mlboard, mlboard)

    elif args.mode == 'CLASSIFY':

        summary_accuracy = 1

        # Classify images
        for algorithm in algorithms:
            print_fun('Testing classifier %s' % algorithm)
            classifier_filename = get_classifier_path(classifiers_path, algorithm)
            with open(classifier_filename, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print_fun('Loaded classifier model from file "%s"' % classifier_filename)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            if isinstance(model, neighbors.KNeighborsClassifier):
                param_name = 'distance'
                # clf_name = "knn"
                (closest_distances, _) = model.kneighbors(emb_array)
                eval_values = closest_distances[:, 0]
            elif isinstance(model, svm.SVC):
                param_name = 'probability'
                # clf_name = "svm"
                eval_values = predictions[np.arange(len(best_class_indices)), best_class_indices]
            else:
                raise RuntimeError("Unsupported classifier type: %s" % type(model))

            for i in range(len(best_class_indices)):
                predicted = best_class_indices[i]
                if predicted == labels[i]:
                    print_fun('%4d  %s: %s %.3f' % (
                        i, class_names[predicted], param_name, eval_values[i],
                    ))
                else:
                    print_fun('%4d  %s: %s %.3f, WRONG! Should be %s.' % (
                        i, class_names[predicted], param_name, eval_values[i], class_names[labels[i]]),
                              )

            accuracy = np.mean(np.equal(best_class_indices, labels))
            summary_accuracy = min(summary_accuracy, accuracy)

            rpt = confusion(labels, best_class_indices, class_names,
                            use_mlboard and not args.skip_draw_confusion_matrix)
            data = {
                'accuracy': accuracy,
                # 'average_time': '%.3fms' % average_time
            }
            if not args.skip_draw_confusion_matrix:
                data['#documents.confusion_matrix.html'] = rpt
            update_data(data, use_mlboard, mlboard)

            print_fun('Accuracy for %s: %.3f' % (algorithm, accuracy))

        if args.upload_model and summary_accuracy >= args.upload_threshold:
            timestamp = datetime.datetime.now().strftime('%s')
            model_name = 'facenet-classifier'

            if args.device == 'MYRIAD':
                model_name = model_name + "-movidius"

            version = '1.0.0-%s-%s' % (args.driver, timestamp)

            print_fun('Uploading model as %s:%s' % (model_name, version))
            upload_model(
                use_mlboard,
                mlboard,
                classifiers_path,
                model_name,
                version
            )


def get_classifier_path(classifiers_path, algorithm):
    return os.path.join(classifiers_path, "classifier-%s.pkl" % algorithm.lower())


def embeddings_per_path(noise_count=0, flip=False):
    # each image with noise count multiplied by 2 if flipped (for each - original and noised)
    return (1 + noise_count) * (2 if flip else 1)


def load_data(paths_batch, labels, image_size=160, noise_count=0, flip=False):
    if len(paths_batch) != len(labels):
        raise RuntimeError("load_data: len(paths_batch) = %d != len(labels) = %d", len(paths_batch), len(labels))

    init_batch_len = len(paths_batch)

    images = facenet.load_data(paths_batch, False, False, image_size)
    images_size = len(images)

    if flip:
        for k in range(images_size):
            img = images[k]
            # print_fun('Applying flip to image {}'.format(paths_batch[k]))
            flipped = facenet.horizontal_flip(img)
            images = np.concatenate((images, flipped.reshape(1, *flipped.shape)))
            labels.append(labels[k])
            paths_batch.append(paths_batch[k])

    if noise_count > 0:
        for k in range(images_size):
            img = images[k]
            for i in range(noise_count):
                # print_fun('Applying noise to image {}, #{}'.format(paths_batch[k], i + 1))
                noised = facenet.random_noise(img)
                images = np.concatenate((images, noised.reshape(1, *noised.shape)))
                labels.append(labels[k])
                paths_batch.append(paths_batch[k])

                if flip:
                    # print_fun('Applying flip to noised image {}, #{}'.format(paths_batch[k], i + 1))
                    flipped = facenet.horizontal_flip(noised)
                    images = np.concatenate((images, flipped.reshape(1, *flipped.shape)))
                    labels.append(labels[k])
                    paths_batch.append(paths_batch[k])

    batch_log = ' ... %d images' % len(images)
    if noise_count > 0 or flip:
        batch_log_details = ['%d original' % init_batch_len]
        if noise_count > 0:
            batch_log_details.append('%d noise' % (init_batch_len * noise_count))
        if flip:
            batch_log_details.append('%d flip' % init_batch_len)
        if noise_count > 0 and flip:
            batch_log_details.append('%d noise+flip' % (init_batch_len * noise_count))
        batch_log = '%s (%s)' % (batch_log, ', '.join(batch_log_details))
    print_fun(batch_log)

    return images


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
        '--classifiers_path',
        help='Path to classifier models stored as pickle (.pkl) files. ' +
             'For training this files are the output and for classification this are an input.',
        required=True,
    )
    parser.add_argument(
        '--only_algorithm',
        help='Train only specified classifier.',
        default=None,
        choices=["SVM", "kNN"],
    )
    parser.add_argument(
        '--knn-neighbors',
        help='Neighbors count (only for kNN classifier).',
        type=int,
        default=1,
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
        '--flip',
        action='store_true',
        help='Add horizontal flip to images.',
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
    parser.add_argument(
        '--skip-draw-confusion-matrix',
        action='store_true',
        default=False,
    )

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
