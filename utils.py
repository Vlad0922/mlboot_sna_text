# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import torch

import os
import glob
import io

import tqdm

import random


def get_subdirs(path):
    return [os.path.join(path, subdir_path) for subdir_path in list(os.walk(path))[0][1]]


def read_metadata(base_path):
    subdirs = get_subdirs(base_path)

    df_list = list()

    for curr_dir in subdirs:
        fname_pattern = os.path.join(curr_dir, '*.parquet')
        parquet_files = list(glob.glob(fname_pattern))
        for fname in parquet_files:
            metadata = pd.read_parquet(fname, engine='fastparquet')
            df_list.append(metadata)

    return pd.concat(df_list)


def read_texts(base_path, disable_tqdm=True, tqdm_module=tqdm.tqdm):
    df_list = list()

    fname_pattern = os.path.join(base_path, '*.parquet')
    parquet_files = list(glob.glob(fname_pattern))
    for fname in tqdm_module(parquet_files, disable=disable_tqdm):
        texts = pd.read_parquet(fname, engine='fastparquet')
        df_list.append(texts)

    return pd.concat(df_list)


def convert_single(word):
    if type(word) is bytes:
        return word.decode('utf-8')
    else:
        return word


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_vectors(fname, disable_tqdm=True, tqdm_module=tqdm.tqdm):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    word2index = dict()
    embeddings = np.zeros((n, d))

    for idx, line in tqdm_module(enumerate(fin), total=n, disable=disable_tqdm):
        tokens = line.rstrip().split(' ')
        word2index[tokens[0]] = idx
        embeddings[idx] = np.asarray(tokens[1:], dtype=np.float32)

    return embeddings, word2index