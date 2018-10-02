##
#  Runs align -> train -> validate tasks.
#  Validate task also may upload the model.
##
import logging
import sys

from mlboardclient.api import client


SUCCEEDED = 'Succeeded'

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
mlboard = client.Client()
run_tasks = ['align-images', 'train-classifier', 'validate-classifier']


def main():
    app = mlboard.apps.get()

    for task in run_tasks:
        t = app.tasks.get(task)

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
