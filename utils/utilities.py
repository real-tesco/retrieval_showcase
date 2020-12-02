import streamlit as st
from pyserini.search import SimpleSearcher
import time


def rerank(hits, ranker, formatter):
    # calculate new score for every hit and query
    pass


def format_retrieved_doc(search_result, shortened):
    if shortened:
        length = min(1000, len(search_result[1]))
    else:
        length = len(search_result[1])

    return '<br/><div style="font-family: Times New Roman; font-size: 20px;''padding-bottom:12px"><b>Score</b>: ' + \
           str(search_result[2]) + '<br><b>Document: ' + search_result[0] + ' </b><br> ' + search_result[1][:length] + '</div>'


def show_query_results(hits, shortened, show_k=2):
    """HTML print format for the searched query"""
    for i, hit in enumerate(hits):
        st.write(format_retrieved_doc(hit, shortened), unsafe_allow_html=True)


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
        for label, distance in zip(labels, distances):
            self._docids.append(label)
            self._doc_scores.append(distance)
            self._doc_content.append(self._index.doc(label))
            self._doc_embeddings.append(self._index.doc(label))
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

