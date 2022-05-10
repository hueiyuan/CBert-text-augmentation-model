import os
import re
import sys
import time
import traceback
from yaml import Loader, load

CONF_SUFFIX = 'config.yaml'

class config():
    def __init__(self, service_name, config_folder_prefix, config_prefix_name, platform=None):
        self.CONF_NAME = config_prefix_name + '_' + CONF_SUFFIX
        pattern = re.compile('.+({})\/\w+'.format(service_name), flags=re.I)
        file_path = os.path.abspath(__file__)
        result = pattern.search(file_path)
        if result:
            config_path = os.path.dirname(result.group(0))
            self.CONF_PATH = os.path.join(config_path, '{}/{}'.format(
                config_folder_prefix, self.CONF_NAME))
        else:
            self.HOME = os.getcwd()
            file_path = os.path.abspath(__file__)
            if 'DATAPROC_VERSION' in os.environ.keys() or platform == 'AWS':
                print('[INFO] SPARK')
                self.CONF_PATH = os.path.join(self.HOME, self.CONF_NAME)
            else:
                print('[INFO] ML-ENGINE / LOCAL')
                self.CONF_PATH = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(file_path))),
                    '{}/{}'.format(config_folder_prefix, self.CONF_NAME)
                    )
        self.org_mtime = time.ctime(os.path.getmtime(self.CONF_PATH))
        try:
            self._load()
        except Exception as e:
            sys.stderr.write('Error loading config "%s": %s\n' %
                             (self.CONF_PATH, e))
            traceback.print_exc(file=sys.stderr)
            exit(1)

    def _load(self):
        stream = open(self.CONF_PATH, 'r')
        config = load(stream, Loader=Loader)
        return config

    def reload(self):
        mtime = time.ctime(os.path.getmtime(self.CONF_PATH))
        if mtime > self.org_mtime:
            try:
                self._load()
                print('New config load')
                self.org_mtime = mtime
            except Exception as e:
                sys.stderr.write(
                    'Error reloading config, keep original version: %s\n' % (e))
                self.org_mtime = mtime
