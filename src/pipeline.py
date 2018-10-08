##
#  Runs align -> train -> validate tasks.
#  Validate task also may upload the model.
##
import argparse
import logging
import re
import sys

from mlboardclient.api import client


SUCCEEDED = 'Succeeded'

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
mlboard = client.Client()
run_tasks = ['align-images', 'train-classifier', 'validate-classifier']


def override_task_arguments(task, params):
    for k, v in params.items():
        pattern = re.compile('{}[ =]([^\s]+|[\'"].*?[\'"])'.format(k))
        resource = task.config['resources'][0]
        task_cmd = resource['command']
        replacement = '{} {}'.format(k, v)
        if pattern.findall(task_cmd):
            # Replace
            resource['command'] = pattern.sub(
                replacement,
                task_cmd
            )
        else:
            # Add
            if 'args' in resource:
                resource['args'][k] = v
            else:
                resource['args'] = {k: v}


def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('upload_threshold')
    parser.add_argument('driver')
    parser.add_argument('--convert', type=boolean_string, default=False)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    override_args = {
        'validate-classifier': {
            'upload-threshold': args.upload_threshold,
            'driver': args.driver,
        },
        'train-classifier': {
            'driver': args.driver,
        }
    }

    if args.convert:
        # Convert model before use it
        run_tasks.insert(0, 'model-converter')
    else:
        # Use converted model
        override_args['train-classifier']['$TRAINING_DIR/faces_160'] = '$MODEL_DIR/facenet.xml'
        override_args['validate-classifier']['$TRAINING_DIR/faces_160'] = '$MODEL_DIR/facenet.xml'

    app = mlboard.apps.get()

    for task in run_tasks:
        t = app.tasks.get(task)
        if t.name in override_args and override_args[t.name]:
            override_task_arguments(t, override_args[t.name])

        LOG.info("Start task %s..." % t.name)
        started = t.start()

        LOG.info(
            "Run & wait [name=%s, build=%s, status=%s]"
            % (started.name, started.build, started.status)
        )
        completed = started.wait()

        if completed.status != SUCCEEDED:
            LOG.warning(
                "Task %s-%s completed with status %s."
                % (completed.name, completed.build, completed.status)
            )
            LOG.warning(
                'Please take a look at the corresponding task logs'
                ' for more information about failure.'
            )
            LOG.warning("Workflow completed with status ERROR")
            sys.exit(1)

        LOG.info(
            "Task %s-%s completed with status %s."
            % (completed.name, completed.build, completed.status)
        )

    LOG.info("Workflow completed with status SUCCESS")


if __name__ == '__main__':
    main()
