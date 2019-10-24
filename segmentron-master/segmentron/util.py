import logging
import os
import shutil
import sys

import click


def setup_output(output):
    if os.path.isdir(output):
        click.confirm(click.style(f"{output} already exists. Do you want to "
            "obliterate it and continue?", fg='red'),
            abort=True)
        shutil.rmtree(output)
    try:
        os.makedirs(output, exist_ok=True)
    except:
        raise Exception(f"Could not create experiment dir {output}")


def setup_logging(logfile):
    FORMAT = '[%(asctime)s.%(msecs)03d] %(message)s'
    DATEFMT = '%Y-%m-%d %H:%M:%S'
    logging.root.handlers = []  # clear logging from elsewhere
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFMT,
            handlers=[
                logging.FileHandler(logfile),
                logging.StreamHandler(sys.stdout)
            ])
    logger = logging.getLogger()
    return logger
