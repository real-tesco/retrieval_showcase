import nltk

class SnippetGenerator:
    def __init__(self, bm25):
        bm25 = bm25

    def __call__(self, *args, **kwargs):
        sentences = args[0].split(".|!|?|")
        query = args[1]
        best_sents = []
        best_sents_ids = []
        for sent in sentences:
