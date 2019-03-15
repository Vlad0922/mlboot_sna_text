# -*- coding: utf-8 -*-

import torch

from sklearn.model_selection import StratifiedKFold

from glove import Corpus, Glove

import gc

from utils import *
from features import *
from trainer import *

from datasets import LikesFeaturesDataset
from networks import NeuralNetFeaturesConv, NeuralNetFeatures
from clr import CyclicLR


seed_everything(42)

input_path = 'input/'

TRAIN_TEXTS = os.path.join(input_path, 'texts', 'textsTrain')
TRAIN_META = os.path.join(input_path, 'textsTrain')

glove = Glove.load('glove_300_mc4_ink_w10.model')
word2index = glove.dictionary
embeddings = glove.word_vectors

max_seq_len = 64
embed_size = embeddings.shape[1]

meta = read_metadata(TRAIN_META)
texts = read_texts(TRAIN_TEXTS, disable_tqdm=True)

bad_objects = get_bad_logs(texts)
texts = texts[[not(v in bad_objects) for v in texts.objectId.values]]
meta = meta[[not(v in bad_objects) for v in meta.instanceId_objectId.values]]

meta['audit_timestamp'] = pd.to_datetime(meta['audit_timestamp'], unit='ms')
meta['audit_hour'] = [v.hour for v in meta.audit_timestamp]

custom_onehot(meta, 'instanceId_objectType')
custom_onehot(meta, 'audit_clientType')
custom_onehot(meta, 'audit_hour')

meta['liked'] = [float('Liked' in v) for v in meta.feedback.values]

texts['preprocessed'] = [np.array([convert_single(w) for w in seq]) for seq in texts['preprocessed'].values]

texts['preprocessed_idx'] = [get_embedding_indexes(word2index, seq, unknown_token='<unk>') for seq in texts.preprocessed.values]

texts['text_len'] = [len(seq) for seq in texts.text]
texts['token_num'] = [len(seq) for seq in texts.preprocessed]

texts.set_index('objectId', inplace=True)

meta.rename(columns={'instanceId_objectId': 'objectId'}, inplace=True)
meta = meta.join(texts, on='objectId')

stat_features = ['instanceId_objectType_Photo', 'instanceId_objectType_Post', 'instanceId_objectType_Video',
                'audit_clientType_API', 'audit_clientType_MOB', 'audit_clientType_WEB',
                'audit_hour_0', 'audit_hour_1','audit_hour_2', 'audit_hour_3', 'audit_hour_4', 'audit_hour_5',
                'audit_hour_6', 'audit_hour_7', 'audit_hour_8', 'audit_hour_9','audit_hour_10', 'audit_hour_11',
                'audit_hour_12', 'audit_hour_13', 'audit_hour_14', 'audit_hour_15', 'audit_hour_16', 'audit_hour_17',
                'audit_hour_18', 'audit_hour_19', 'audit_hour_20', 'audit_hour_21', 'audit_hour_22', 'audit_hour_23',
                'text_len', 'token_num']


try:
    model = NeuralNetFeatures(embeddings, hidden_size=30, n_features=len(stat_features), train_embed=False)
    model.cuda()
except Exception as e:
    del model
    gc.collect()

skf = StratifiedKFold(n_splits=5, random_state=42)

for fold_idx, (train_index, test_index) in enumerate(skf.split(meta, meta.liked)):
    print('Training model for fold: {}'.format(fold_idx))
    data_train = meta.iloc[train_index]

    batch_size = 512 * 4
    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_dataset = LikesFeaturesDataset(data_train, stat_features, max_seq_len, idx_col='preprocessed_idx')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    model = NeuralNetFeaturesConv(embeddings, max_seq_len=max_seq_len, embed_size=embed_size,
                                  hidden_size=64, n_features=len(stat_features), train_embed=False)
    model.cuda()

    loss_fn = torch.nn.BCELoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CyclicLR(optimizer, step_size=10 * len(train_loader), mode='exp_range', gamma=0.99999)
    n_epochs = 20

    for epoch in tqdm.tnrange(n_epochs):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, scheduler)

    model_fname = 'mdl_gru_conv_fold_{}.pth'.format(fold_idx)
    model_path = os.path.join('models', model_fname)

    torch.save(model.state_dict(), model_path)

    del model
    del data_train

    gc.collect()
