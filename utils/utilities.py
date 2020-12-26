import streamlit as st
from pyserini.search import SimpleSearcher
import time
from models.retriever.knn_retriever import KnnIndex
import logging
import torch
import re

logger = logging.getLogger()


def rerank(q_embedding, doc_embeddings, hits, ranker):
    device = torch.device('cpu')
    scores = ranker.rerank_documents(q_embedding, doc_embeddings.squeeze(), device).tolist()
    hits = [(hit[0], hit[1], score, doc_embed) for hit, score, doc_embed in zip(hits, scores, doc_embeddings)]
    hits.sort(key=lambda tup: tup[2], reverse=True)
    return hits


def reformulate(q_embedding, hits, reformulator, reformulator_type):
    doc_embeddings = torch.tensor([hit[3].tolist() for hit in hits])
    if reformulator_type.split(" ")[0] == "Transformer":
        new_query = reformulator(q_embedding, doc_embeddings)
    elif reformulator_type.split(" ")[0] == 'Neural':
        new_query = reformulator(q_embedding, doc_embeddings)
    elif reformulator_type.split(" ")[0] == "Weighted":
        scores = torch.tensor([hit[2] for hit in hits])
        new_query = reformulator(doc_embeddings, scores)
    return new_query

@st.cache(allow_output_mutation=True)
def load_knn_index(args):
    logger.info("Cache miss: load_index_file() runs")
    knn_index = KnnIndex(args)
    checkpoint = torch.load(args.two_tower_checkpoint, map_location=torch.device('cpu'))
    knn_index.load_state_dict(checkpoint)
    knn_index.load_index_file()
    return knn_index


@st.cache
def load_index_file(index):
    logger.info("Cache miss: load_index_file() runs")
    index.load_index_file()
    logger.info("loading of index file finished.")
    return index


def format_retrieved_doc(rank, search_result, shortened):
    if shortened:
        length = min(1000, len(search_result[1]))
    else:
        length = len(search_result[1])
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|$'
    source_link = re.findall(pattern, search_result[1][:length])[0]
    return '<br/><div style="font-family: Times New Roman; font-size: 19px;''padding-bottom:12px"><b>Rank: ' + \
           str(rank) + ' Score: ' + str(search_result[2]) + '<br>Document: ' + search_result[0] + ' </b><br> ' \
           + search_result[1][:length].replace(source_link, f"<a href="f"{source_link}"">Source</a>") + '</div>'


def show_query_results(hits, shortened, col):
    """HTML print format for the searched query"""
    with col:
        for i, hit in enumerate(hits):
            st.write(format_retrieved_doc(i+1, hit, shortened), unsafe_allow_html=True)


class SearchResultFormatter:
    def __init__(self, index):
        self._index = SimpleSearcher(index)
        self._query = None
        self._docids = []
        self._doc_content = []
        self._doc_scores = []
        self._doc_embeddings = []

    def pyserini_search_result(self, hits, query):
        self.clear()
        for hit in hits:
            self._docids.append(hit.docid)
            self._doc_scores.append(hit.score)
            self._doc_content.append(hit.raw)
        return query, zip(self._docids, self._doc_content, self._doc_scores), None

    def hnswlib_search_result(self, labels, distances, query_embedding, doc_embeddings):
        self.clear()
        # since only one query at a time:
        labels = labels[0]
        distances = distances[0]
        for label, distance in zip(labels, distances):
            self._docids.append(label)
            self._doc_scores.append(1.0 - distance)
            self._doc_content.append(self._index.doc(label).get('raw'))
            #self._doc_embeddings.append(doc_embeddings)
        return query_embedding, zip(self._docids, self._doc_content, self._doc_scores), doc_embeddings

    def clear(self):
        self._docids = []
        self._doc_content = []
        self._doc_scores = []
        self._doc_embeddings = []
        self._query = None


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total

