from pyserini.search import SimpleSearcher


class BM25Retriever:
    def __init__(self, dataset):
        self.searcher = SimpleSearcher(dataset)
        self.searcher.set_bm25(0.9, 0.4)
        self.searcher.set_rm3(10, 10, 0.5)

    def query(self, query_text):
        hits = self.searcher.search(query_text)
        return hits
