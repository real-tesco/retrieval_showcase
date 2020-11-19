import hnswlib
import torch
import torch.nn as nn
import logging
from twotowerbert import TwoTowerBert
import copy

logger = logging.getLogger()


class KnnIndex:
    def __init__(self, args):
        self.args = args
        self.index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
        self.model = TwoTowerBert(args.pretrained)

    def load_index(self):
        logger.info('Loading KNN index...')
        self.index.load_index(self.args.index_file)

