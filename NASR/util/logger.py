import yaml
import logging
import logging.config
import os


def init_logger():
    file_dir = os.path.dirname(__file__)
    fpath = os.path.join(file_dir, 'logging.yaml')
    with open(fpath) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    logging.config.dictConfig(config)
    return logging.getLogger('__main__')
