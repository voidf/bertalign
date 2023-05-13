import datetime
from difflib import SequenceMatcher
import itertools
import os

from pathlib import Path
import traceback
from datasets import load_dataset, load_from_disk
from bertalign import Bertalign
import process_zh_text
import process_en_text
import numpy as np
import json
from collections import OrderedDict, Counter
import re
from datasets.dataset_dict import DatasetDict
##
from sentence_transformers import SentenceTransformer
import time

from helper import dump_row, ensure_dirs, make_filter_log, my_path, align_logger, use_proxy
from helper import PAGINATION_TOKEN, LANGS
from helper import dump_align_result_to_file
from helper import PREPROCESS_DIR, ALIGNED_DIR, FILTER_LOG, ERROR_LOG
from helper import chat

LINE_STEP = 20

def process_one_file_use_chatgpt(row: DatasetDict):
    inputs = row['en'].splitlines()
    rec = row['record']

    for i in range(0, len(inputs), LINE_STEP):
        input_batch = '\n'.join(inputs[i : i + 2 * LINE_STEP])
        # done = False
        for retrytime in range(3):
            try:
                outputs = chat(input_batch)
            except Exception as e:
                print(e)
                print('sleep for 10s')
                time.sleep(10)

                if retrytime == 2:
                    raise

        with open(my_path(f'done/gpt_en_{rec}.txt'), 'a', encoding='utf-8') as f:
            f.write(outputs + PAGINATION_TOKEN)

        print('sleep for 10s')
        time.sleep(10)


if __name__ == "__main__":
    dataset = load_dataset('bot-yaya/UN_PDF_SUBSET_PREPROCESSED')
    cmd = input('use proxy? (default settings is socks5://localhost:7890) please answer(y/N):')
    if cmd.lower() == 'y':
        use_proxy()
    
    dataset.map(process_one_file_use_chatgpt)
