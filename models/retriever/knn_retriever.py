import hnswlib
import torch
import logging
from models.retriever.two_tower_bert import TwoTowerBert
from transformers import AutoTokenizer
from utils.utilities import SearchResultFormatter
import json

logger = logging.getLogger()


class KnnIndex:
    def __init__(self, args):
        self._args = args

        self._model = TwoTowerBert(pretrained=args.two_tower_base)
        self._tokenizer = AutoTokenizer.from_pretrained(args.two_tower_base)
        if args.two_tower_checkpoint is not None:
            logger.info("Loading checkpoint...")
            state_dict = torch.load(args.two_tower_checkpoint, map_location=torch.device('cpu'))
            self._model.load_state_dict(state_dict)
        else:
            logger.info("no checkpoint found...")
        logger.info('Loading index file')
        self._index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
        self._index.load_index(args.index_file)
        #self._formatter = formatter
        self._seq_max_len = args.max_query_len_input
        self._docid2indexid = {}
        self._indexid2docid = {}

        with open(args.index_mapping, 'r') as f:
            mapping = json.load(f)
        for key in mapping:
            self._indexid2docid[mapping[key]] = key
            self._docid2indexid[key] = mapping[key]

    def set_formatter(self, formatter):
        self._formatter = formatter

    def query(self, query_text, k=100):
        query_tokens = self._tokenizer.tokenize(query_text)[:self._seq_max_len - 2]
        input_ids, segment_ids, input_mask = self.pack_bert_features(query_tokens)
        query_embedding = self._model.calculate_embedding(input_ids, segment_ids, input_mask, doc=False)

        labels, distances = self._index.knn_query(query_embedding.detach().numpy(), k=k)
        distances = distances.tolist()
        labels = labels[0].tolist()
        document_labels = [self._indexid2docid[labels[i]] for i in range(len(labels))]
        print(document_labels)
        result = self._formatter.hnswlib_search_result(document_labels, distances, query_embedding, self._index.get_items(labels))
        return result

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

