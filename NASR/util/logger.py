import yaml
import logging
import logging.config
import os


def init_logger():
    file_dir = os.path.dirname(__file__)
    f_path = os.path.join(file_dir, 'config/logging.yaml')
    with open(f_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    logging.config.dictConfig(config)
    return logging.getLogger('__main__')
