# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

import numpy as np

class BaselineDataset(Dataset):
    def __init__(self, meta, max_seq_len, is_test=False):
        self.meta = meta
        self.max_seq_len = max_seq_len
        self.is_test = is_test

        self.texts = meta.preprocessed_idx.values

        if not(self.is_test):
            self.likes = meta.liked.values.astype(float)

    def _pad_with_zeros(self, seq):
        diff = self.max_len - len(seq)
        if diff > 0:
            return np.concatenate((np.zeros(diff, dtype=int), seq))
        else:
            return np.array(seq[:self.max_len])

    def __getitem__(self, index):
        if self.is_test:
            target = np.float32(0.0)
        else:
            target = self.likes[index]

        return self._pad_with_zeros(self.texts[index]).astype(int), target

    def __len__(self):
        return len(self.texts)


class LikesReshareDataset(Dataset):
    def __init__(self, meta, max_seq_len, idx_col='preprocessed_idx', is_test=False):
        self.meta = meta
        self.max_len = max_seq_len
        self.is_test = is_test

        self.texts = meta[idx_col].values

        if not(is_test):
            self.likes = meta.liked.values.astype(float)
            self.repost = meta.reshared.values.astype(float)

    def _pad_with_zeros(self, seq):
        diff = self.max_len - len(seq)
        if diff > 0:
            return np.concatenate((np.zeros(diff, dtype=int), seq))
        else:
            return np.array(seq[:self.max_len], dtype=int)

    def __getitem__(self, index):
        if self.is_test:
            target = (np.float32(0.0), np.float32(0.0))
        else:
            target = (self.likes[index], self.repost[index])

        return self._pad_with_zeros(self.texts[index]).astype(int), target

    def __len__(self):
        return len(self.texts)


class LikesFeaturesDataset(Dataset):
    def __init__(self, meta, features, max_len, idx_col='preprocessed_idx', is_test=False):
        self.meta = meta
        self.max_len = max_len
        self.is_test = is_test

        self.features = self.meta[features].values.astype(np.float32)

        self.texts = meta[idx_col].values

        if not(is_test):
            self.likes = meta.liked.values.astype(np.float32)

    def _pad_with_zeros(self, seq):
        diff = self.max_len - len(seq)
        if diff > 0:
            return np.concatenate((np.zeros(diff, dtype=int), seq))
        else:
            return np.array(seq[:self.max_len], dtype=int)

    def __getitem__(self, index):
        seq = self._pad_with_zeros(self.texts[index]).astype(int)
        f = self.features[index]

        if self.is_test:
            target = np.float32(0.0)
        else:
            target = self.likes[index]

        return (seq, f), target

    def __len__(self):
        return len(self.texts)


class LikesFeaturesTailDataset(Dataset):
    def __init__(self, meta, embeddings, features, max_len, idx_col='preprocessed_idx', is_test=False):
        self.meta = meta
        self.max_len = max_len
        self.embeddings = embeddings
        self.is_test = is_test

        self.features = self.meta[features].values.astype(np.float32)

        self.texts = meta[idx_col].values

        if not(is_test):
            self.likes = meta.liked.values.astype(np.float32)

    def _pad_with_zeros(self, seq):
        diff = self.max_len - len(seq)
        if diff > 0:
            return np.concatenate((np.zeros(diff, dtype=int), seq))
        else:
            return np.array(seq[:self.max_len], dtype=int)

    def _get_tail(self, seq):
        tail_idx = seq[self.max_len:]

        if len(tail_idx) > 0:
            return self.embeddings[tail_idx].mean(axis=0)
        else:
            return np.zeros(self.embeddings.shape[1], dtype=np.float32)

    def __getitem__(self, index):
        seq = self._pad_with_zeros(self.texts[index]).astype(int)
        f = self.features[index]
        tail = self._get_tail(self.texts[index]).astype(np.float32)

        if self.is_test:
            target = np.float32(0.0)
        else:
            target = self.likes[index]

        return (seq, f, tail), target

    def __len__(self):
        return len(self.texts)