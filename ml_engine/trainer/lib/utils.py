import re
import argparse
import sys
import pickle
import subprocess
import warnings
import logging
from datetime import datetime, timedelta
from retrying import retry


def get_period_of_date(current_date, period_days):
    end_date = None
    start_date = None
    if isinstance(current_date, str):
        end_date = datetime.strptime(
            re.sub('\D', '-', current_date), "%Y-%m-%d")
        start_date = end_date - timedelta(days=period_days) + timedelta(days=1)
    elif isinstance(current_date, datetime):
        end_date = current_date
        start_date = end_date - timedelta(days=period_days) + timedelta(days=1)
    else:
        print(
            'Date format erroy, please use datetime or string[%Y-%m-%d] format')
    return start_date, end_date


def transfer_data_by_gsutil(input_path, output_path, is_folder=False):
    logging.info('[INFO] Transfer data from {} to {}'.format(input_path, output_path))
    try:
        if not is_folder:
            cmd = 'gsutil -m cp {} {}'.format(input_path, output_path)
            subprocess.check_output(cmd, shell=True)
        else:
            cmd = 'gsutil -m cp -r {} {}'.format(input_path, output_path)
            subprocess.check_output(cmd, shell=True)
    except Exception as e:
        sys.exit(IOError('[ERROR] {}'.format(e)))


def transfer_data_by_awscli(input_path, output_path, is_folder=False):
    logging.info('[INFO] Transfer data from {} to {}'.format(input_path, output_path))
    try:
        if not is_folder:
            cmd = 'aws s3 cp {} {}'.format(input_path, output_path)
        else:
            cmd = 'aws s3 cp {} {} --recursive'.format(input_path, output_path)
        subprocess.check_output(cmd, shell=True)
    except Exception as e:
        sys.exit(IOError('[ERROR] {}'.format(e)))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class StoreDictKeyPair(argparse.Action):
    # https://stackoverflow.com/questions/29986185/python-argparse-dict-arg
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def save_data_to_pkl(data, output_file):
    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(input_file):
    with open(str(input_file), 'rb') as handle:
        data = pickle.load(handle)
    return data
