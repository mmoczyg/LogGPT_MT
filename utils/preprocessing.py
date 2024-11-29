import os
import tarfile
import urllib.request
import shutil
import json
import pandas as pd
from . import drain, drainTB



# Drain: https://github.com/logpai/logparser

def parsing(dataset_name, output_dir='./datasets/'):
    """Download and parsing dataset

    Args:
        dataset_name: name of the log dataset
        output_dir: directory name for datasets storage

    Returns:
        Structured log datasets in Pandas Dataframe after adopt Drain
    """
    path = os.getcwd()
    directory = path + output_dir[1:]
    if not os.path.exists(directory):
        print(f'Making directory for dataset storage {directory}')
        os.makedirs(directory)

    if dataset_name == 'MtController':
        # Since the dataset is already provided, no need to download
        input_dir = '/home/mmoczyg/LogGPT/datasets'  # The input directory of log file
        log_file = 'mt_controller_log.json'  # The input log file name
        log_format = '<date> <time> <session_id> <event_id> <severity> <reporter> <msg>'  # MtController log format
        # Regular expression list for optional preprocessing (default: [])
        regex = [r'']
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes

        parser = drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_file)

	 # Save the parsed log to a structured CSV
        structured_log_path = os.path.join(output_dir, 'MtController_structured.csv')
        parser.df_log.to_csv(structured_log_path, index=False)

        try:
            os.remove(log_file)
        except OSError:
            pass

    else:
        raise ValueError('Invalid dataset name')
