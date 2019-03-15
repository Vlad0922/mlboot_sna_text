# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from net_layers import Attention


class NeuralNetBaseline(nn.Module):
    def __init__(self, embeddings, max_seq_len, embed_size, hidden_size=30, n_out=1, train_embed=False):
        super(NeuralNetBaseline, self).__init__()

        self.embedding = nn.Embedding(embeddings.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float32))
        self.embedding.weight.requires_grad = train_embed

        self.embedding_dropout = nn.Dropout2d(0.2)

        self.hidden_size = hidden_size

        self.conc_size = self.hidden_size * 8

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, max_seq_len)
        self.gru_attention = Attention(hidden_size * 2, max_seq_len)

        self.linear = nn.Linear(self.conc_size, 64)
        self.relu = nn.PReLU()

        self.bn = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(64, n_out)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)

        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.bn(conc)

        out = torch.sigmoid(self.out(conc))

        return out


class NeuralNetFeatures(nn.Module):
    def __init__(self, embeddings, max_seq_len, embed_size,  n_features=1, hidden_size=30, train_embed=False,
                 embed_dtype=torch.float32):
        super(NeuralNetFeatures, self).__init__()

        self.embedding = nn.Embedding(embeddings.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=embed_dtype))
        self.embedding.weight.requires_grad = train_embed

        self.embedding_dropout = nn.Dropout2d(0.2)

        self.hidden_size = hidden_size

        self.conc_size = self.hidden_size * 8

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, max_seq_len)
        self.gru_attention = Attention(hidden_size * 2, max_seq_len)

        self.linear = nn.Linear(self.conc_size + n_features, 64)
        self.relu = nn.PReLU()

        self.bn = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(64, 1)

    def forward(self, seq, features):
        h_embedding = self.embedding(seq)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0))).type(torch.float32)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, features), 1)

        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.bn(conc)

        out = torch.sigmoid(self.out(conc))

        return out


class NeuralNetFeaturesTail(nn.Module):
    def __init__(self, embeddings, max_seq_len, embed_size, n_features=1, hidden_size=30,  train_embed=False):
        super(NeuralNetFeaturesTail, self).__init__()

        self.embedding = nn.Embedding(embeddings.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float32))
        self.embedding.weight.requires_grad = train_embed

        self.embedding_dropout = nn.Dropout2d(0.2)

        self.hidden_size = hidden_size

        self.conc_size = self.hidden_size * 8

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, max_seq_len + 1)
        self.gru_attention = Attention(hidden_size * 2, max_seq_len + 1)

        self.linear = nn.Linear(self.conc_size + n_features, 64)
        self.relu = nn.PReLU()

        self.bn = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(64, 1)

    def forward(self, seq, features, tail):
        h_embedding = self.embedding(seq)
        h_embedding = torch.cat((h_embedding, tail.unsqueeze(1)), dim=1)

        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, features), 1)

        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.bn(conc)

        out = torch.sigmoid(self.out(conc))

        return out


class NeuralNetFeaturesConv(nn.Module):
    def __init__(self, embeddings, max_seq_len, embed_size, hidden_size=64, n_features=1, train_embed=False,
                 embed_dtype=torch.float32, ks=3, kn=64):
        super(NeuralNetFeaturesConv, self).__init__()

        self.embedding = nn.Embedding(embeddings.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=embed_dtype))
        self.embedding.weight.requires_grad = train_embed

        self.embedding_dropout = nn.Dropout2d(0.2)

        self.hidden_size = hidden_size

        self.rnn = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.rnn_conv = nn.Conv1d(max_seq_len, kn, ks)

        self.conc_size = max_seq_len * 2 - ks + 1 + n_features

        self.linear = nn.Linear(self.conc_size, 32)
        self.relu = nn.PReLU()

        self.bn = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(32, 1)

    def forward(self, seq, features):
        h_embedding = self.embedding(seq)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0))).type(torch.float32)

        h_rnn, _ = self.rnn(h_embedding)
        h_conv = self.rnn_conv(h_rnn)

        max_pool, _ = torch.max(h_conv, 1)

        conc = torch.cat((max_pool, features), 1)

        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.bn(conc)

        out = torch.sigmoid(self.out(conc))

        return out