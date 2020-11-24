import hnswlib
import torch
import logging
from models.retriever.twotowerbert import TwoTowerBert
from transformers import AutoTokenizer
from utils.utilities import SearchResultFormatter

logger = logging.getLogger()


class KnnIndex:
    def __init__(self, args, formatter):
        self._args = args
        self._index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
        self._formatter = formatter
        self._model = TwoTowerBert(args.two_tower_checkpoint)
        self._tokenizer = AutoTokenizer.from_pretrained(args.two_tower_base)
        self._seq_max_len = args.max_query_len_input

    def query(self, query_text, k=100):
        query_tokens = self._tokenizer.tokenize(query_text)[:self._seq_max_len - 2]
        input_ids, segment_ids, input_mask = self.pack_bert_features(query_tokens)
        query_embedding = self._model.calculate_embedding(input_ids,segment_ids, input_mask, doc=False)

        labels, distances = self._index.knn_query(query_embedding, k=k)
        result = self._formatter.hnswlib_search_result(labels, distances, query_embedding, self._index.get_items(labels))
        return result

    def load_model_state_dict(self, checkpoint):
        state_dict = torch.load(checkpoint)
        self._model.load_state_dict(state_dict)

    def load_index(self, index_file):
        self._index.load_index(index_file)

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
        return input_ids, segment_ids, input_mask

