#!/usr/bin/python3

import torch
import torch.nn as nn


class EmbeddingRanker(nn.Module):
    def __init__(self, args):
        super(NeuralRanker, self).__init__()
        self._input = nn.Linear(args.ranker_input, args.ranker_hidden)
        self._h1 = nn.Linear(args.ranker_hidden, args.ranker_hidden)
        self._output = nn.Linear(args.ranker_hidden, 1)

        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(p=0.1)
        self._batchnorm1 = nn.BatchNorm1d(args.ranker_hidden)
        self._batchnorm2 = nn.BatchNorm1d(args.ranker_hidden)

    # no sigmoid since BCE loss with logits
    def forward(self, inputs):
        x = self._input(inputs)
        # x = self.relu(x)
        x = self._batchnorm1(x)
        x = self._h1(x)
        # x = self.relu(x)
        x = self._batchnorm2(x)
        if self.args.train:
            x = self._dropout(x)
        x = self._output(x)

        return x

    def score_documents(self, queries, positives, negatives=None):
        positives = torch.cat((queries, positives), dim=1)
        scores_positive = self.forward(positives)

        if negatives is not None:
            negatives = torch.cat((queries, negatives), dim=1)
            scores_negative = self.forward(negatives)
            return scores_positive, scores_negative

        return scores_positive

