from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer


class TwoTowerBertIndex(nn.Module):
    def __init__(
        self,
        pretrained: str,
        mode: str = 'cls',
        task: str = 'ranking'
    ) -> None:
        super(TwoTowerBertIndex, self).__init__()
        self._pretrained = pretrained
        self._mode = mode
        self._task = task

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._document_model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        self._query_model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained)
        if self._task == 'ranking':
            self._dense = nn.Linear(self._config.hidden_size, 1)
        elif self._task == 'classification':
            self._dense = nn.Linear(self._config.hidden_size, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def query(self, query_text):
        return query_text

    def forward(self, q_input_ids: torch.Tensor, d_input_ids: torch.Tensor, q_input_mask: torch.Tensor = None, q_segment_ids: torch.Tensor = None,
                d_input_mask: torch.Tensor = None, d_segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        query = self._query_model(q_input_ids, attention_mask=q_input_mask, token_type_ids=q_segment_ids)
        document = self._document_model(d_input_ids, attention_mask=d_input_mask, token_type_ids=d_segment_ids)

        #print(query[0].shape)  # [4, 64, 768]
        #print(query[1].shape)  # [4, 768]  #CLS token with linear layer and tanh activation
        query = query[0][:, 0, :]  # CLS Token

        document = document[0][:, 0, :]

        if self._mode == 'cls':
            document = F.normalize(document, p=2, dim=1)
            query = F.normalize(query, p=2, dim=1)

            score = (document * query).sum(dim=1)
            score = torch.clamp(score, min=0.0, max=1.0)

        return score, query, document
