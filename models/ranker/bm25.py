from pyserini.search import SimpleSearcher


class BM25Retriever:
    def __init__(self, dataset):
        #self._formatter = formatter
        self._searcher = SimpleSearcher(dataset)
        self._searcher.set_bm25(0.9, 0.4)
        self._searcher.set_rm3(10, 10, 0.5)

    def query(self, query_text):
        hits = self._searcher.search(query_text)
        result = self._formatter.pyserini_search_result(hits, query_text)
        return result

    def set_formatter(self, formatter):
        self._formatter = formatter
