# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import emoji
import string

import nltk
import jamspell
import pymorphy2

import tqdm

from urlextract import URLExtract

from joblib import Parallel, delayed


LAT_TO_CYR = {'p': 'р', 'a': 'а' , 'c': 'с', 'o': 'о', 'e': 'е', 'm': 'т', 'x': 'х', 'k': 'к'}


def count_emoji(seq):
    return sum([c in emoji.UNICODE_EMOJI for c in seq])


def count_punctuation(seq):
    return sum([c in string.punctuation for c in seq])


def count_upper(seq):
    return sum([c.isupper() for c in seq])


def count_urls_df(df, text_col='text'):
    def _count_single(text):
        try:
            return len(extractor.find_urls(text))
        except Exception as e:
            return 0

    extractor = URLExtract()

    return [_count_single(text) for text in df[text_col].values]


def count_unique_tokens(seq):
    return len(set(seq))


def custom_tokenize(docs, parallel=False, disable_tqdm=True, tqdm_module=tqdm.tqdm):
    tok = nltk.tokenize.TweetTokenizer(preserve_case=False)

    def _tokenize_single(seq):
        return np.array(tok.tokenize(seq))

    if parallel:
        return Parallel(n_jobs=8)(delayed(_tokenize_single)(seq) for seq in tqdm_module(docs, disable=disable_tqdm))
    else:
        return [tok.tokenize(seq) for seq in tqdm_module(docs, disable=disable_tqdm)]


def replace_characters(word):
    return ''.join([LAT_TO_CYR[c] if c in LAT_TO_CYR else c for c in word])


def try_to_find(word, word2index, corrector=None, morph=None):
    idx = word2index.get(word)

    if not (idx is None):
        return idx

    w_fix = replace_characters(word)
    idx = word2index.get(w_fix)

    if not (idx is None):
        return idx

    if not (corrector is None):
        w_corr = corrector.FixFragment(w_fix)
        idx = word2index.get(w_corr)

        if not (idx is None):
            return idx

    if not (morph is None):
        # pymorphy fails on strange symbols like 'square' and I have to catch exception
        try:
            w_normal = morph.normal_forms(w_fix)
            for wn in w_normal:
                idx = word2index.get(wn)
                if not (idx is None):
                    return idx
        except Exception as e:
            pass

    return None


def correct_and_find(df, word2index, token_col='preprocessed', correction=False, normalize=True):
    if correction:
        corrector = jamspell.TSpellCorrector()
        corrector.LoadLangModel('3rdparty/ru_model_subtitles.bin')
    else:
        corrector = None

    if normalize:
        morph = pymorphy2.MorphAnalyzer()
    else:
        morph = None

    def _correct_single(seq):
        seq_idx = list()
        for w in seq:
            idx = try_to_find(w, word2index, corrector, morph)
            if not (idx is None):
                seq_idx.append(idx)

        return seq_idx

    return [_correct_single(seq) for seq in df[token_col].values]


def get_single_embedding(word2index, w, unknown_token):
    res = word2index.get(w)
    if not (res is None):
        return res
    else:
        return word2index[unknown_token]


def get_embedding_indexes(word2index, seq, unknown_token='<unk>'):
    if not(unknown_token is None):
        return [get_single_embedding(word2index, w, unknown_token) for w in seq]
    else:
        return [word2index[w] for w in seq if w in word2index]


# df.join(pd.get_dummies(df[col])) or sklearn onehotencoder consumes too much memory and takes too much time.
def custom_onehot(df, col):
    oh_data = pd.get_dummies(df[col])

    for c in oh_data.columns.values:
        df[col + '_' + str(c)] = oh_data[c]

    del oh_data
