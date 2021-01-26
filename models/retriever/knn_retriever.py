import hnswlib
import torch
import logging
from models.retriever.two_tower_bert import TwoTowerBert
from transformers import AutoTokenizer
import json

logger = logging.getLogger()


class KnnIndex:
    def __init__(self, args):
        self._args = args
        self._model = TwoTowerBert(pretrained=args.two_tower_base)
        self._tokenizer = AutoTokenizer.from_pretrained(args.two_tower_base)
        self._index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
        # self._index.load_index(args.index_file)
        self._seq_max_len = args.max_query_len_input
        self._docid2indexid = {}
        self._indexid2docid = {}

        with open(args.index_mapping, 'r') as f:
            mapping = json.load(f)
        for key in mapping:
            self._indexid2docid[mapping[key]] = key
            self._docid2indexid[key] = mapping[key]

    def query(self, query_text, k=100):
        query_tokens = self._tokenizer.tokenize(query_text)[:self._seq_max_len - 2]
        input_ids, segment_ids, input_mask = self.pack_bert_features(query_tokens)
        query_embedding = self._model.calculate_embedding(input_ids, segment_ids, input_mask, doc=False)

        labels, distances = self._index.knn_query(query_embedding.detach().numpy(), k=k)
        document_labels = [[self._indexid2docid[labels[j][i]] for i in range(len(labels[j]))] for j in
                           range(len(labels))]
        instances = self._index.get_items(labels[0])
        instances = torch.tensor(instances)
        return document_labels, distances.tolist(), query_embedding, instances

    def query_embedded(self, query_embedding, k=100):
        labels, distances = self._index.knn_query(query_embedding.detach().numpy(), k=k)
        document_labels = [[self._indexid2docid[labels[j][i]] for i in range(len(labels[j]))] for j in
                           range(len(labels))]
        instances = self._index.get_items(labels[0])
        instances = torch.tensor(instances)
        return document_labels, distances.tolist(), query_embedding, instances

    def pack_bert_features(self, tokens):
        input_tokens = [self._tokenizer.cls_token] + tokens + [self._tokenizer.sep_token]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [1] * len(input_ids)
        input_mask = [1] * len(input_ids)

        padding_len = self._seq_max_len - len(input_ids)

        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len

        assert len(input_ids) == self._seq_max_len
        assert len(input_mask) == self._seq_max_len
        assert len(segment_ids) == self._seq_max_len
        return torch.tensor(input_ids).view(1, self._seq_max_len), torch.tensor(segment_ids).view(1, self._seq_max_len), torch.tensor(input_mask).view(1, self._seq_max_len)

    def load_index_file(self):
        self._index.load_index(self._args.index_file)

    def load_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def get_document(self, did):
        did = [self._docid2indexid[did]]
        return self._index.get_items(did)[0]
