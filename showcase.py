#!/usr/bin/python3

import streamlit as st
import torch
from utils.config import get_args

# TODO load all models once
import os
from models.retriever.knn_retriever import KnnIndex
from models.ranker.bm25 import BM25Retriever
from models.ranker.ffn_ranker import NeuralRanker
import utils.utilities as utils
import logging

timer = utils.Timer()
logger = logging.getLogger()


def main(args, retrievers, rankers):
    # need setting up of retrievers here since streamlit cannot hash SimpleSearcher
    formatter_dict = {}
    bm25s = {}
    for folder_name in args.possible_bm25_indexes:
        formatter_dict[folder_name] = utils.SearchResultFormatter(os.path.join(args.base_folder_index, folder_name))
        bm25s[folder_name] = BM25Retriever(dataset=os.path.join(args.base_folder_index, folder_name))

    st.title("Information Retrieval Showcase")
    st.write("Showcase for different retriever and ranker architectures")

    # Setting up sidebar with options for retriever, ranker, etc..
    st.sidebar.title("Settings")

    st.sidebar.subheader("Dataset")
    # this needs to effect the retriever and ranker choice
    l_dataset = st.sidebar.selectbox("Select dataset",
                                     ("MSMARCO Doc", "MSMARCO Doc (passaged)", "Robust04"))

    st.sidebar.subheader("Model")
    l_retriever = st.sidebar.selectbox("Select retriever",
                                       ("BM25", "KNN - Two Tower Bert"))
    ranker_possibilities = None
    if l_retriever == "BM25":
        ranker_possibilities = None
    elif l_retriever == "KNN - Two Tower Bert":
        ranker_possibilities = args.possible_rankers

    if ranker_possibilities is not None:
        l_ranker = st.sidebar.selectbox("Select ranker",
                                        tuple(ranker_possibilities))
    else:
        l_ranker = None

    st.sidebar.subheader("Compare Options")
    compare = st.sidebar.checkbox("Compare models")
    if compare:
        l_retriever2 = st.sidebar.selectbox("Select another retriever",
                                            ("BM25", "KNN - Two Tower Bert"))
        l_ranker2 = st.sidebar.selectbox("Select another ranker",
                                         tuple(args.possible_rankers))
    st.sidebar.subheader("Other Options")

    snippets = st.sidebar.checkbox("Show snippets of documents", value=True)
    top_k = st.sidebar.number_input("How many documents to consider", value=100, min_value=1, max_value=1000)

    # Load selected models
    retriever = None
    index_key = None
    if l_dataset == "MSMARCO Doc":
        index_key = "msmarco_anserini_document"
    elif l_dataset == "MSMARCO Doc (passaged)":
        index_key = "msmarco_passaged_150_anserini"
    elif l_dataset == "Robust04":
        index_key = "index-robust04-20191213"

    formatter = formatter_dict[index_key]

    if l_retriever == "BM25":
        retriever = bm25s[index_key]   # BM25Retriever(formatter, path_to_index)
    elif l_retriever == "KNN - Two Tower Bert":
        retriever = retrievers["TwoTowerKNN"]   # KnnIndex(args)

    ranker = None
    if l_ranker == "EmbeddingRanker":
        ranker = rankers["EmbeddingRanker"]

    # Query Input for freestyle exploring
    query = st.text_input("Query", value='')

    timer.reset()
    if l_retriever == "BM25":
        hits, _ = retriever.query(query)
        hits = formatter.pyserini_search_result(hits, query)[1]
    elif l_retriever == "KNN - Two Tower Bert":
        q_embedding, hits, doc_embeddings = formatter.hnswlib_search_result(*retriever.query(query, k=top_k))
    hits = list(hits)
    retriever_time = timer.time()
    st.write(f"Time to find {len(hits)} documents fur current query: {retriever_time}s")

    if l_ranker == "EmbeddingRanker" and l_retriever == "KNN - Two Tower Bert":
        timer.reset()
        hits = utils.rerank(q_embedding, doc_embeddings, hits, ranker)
        reranker_time = timer.time()
        st.write(f"Time to rerank: {reranker_time}")
    utils.show_query_results(hits, shortened=snippets, show_k=10)


@st.cache(allow_output_mutation=True)
def load_models_on_start(args):
    logger.info("load_models_on_start cache miss")
    retrievers = {}
    rankers = {}
    # Load models here
    for index_type in args.possible_knn_indexes:
        if index_type == "TwoTowerKNN":
            logger.info("load two tower index")
            knn_index = utils.load_knn_index(args)
            logger.info("loaded full index")
            #index = KnnIndex(args)
            #checkpoint = torch.load(args.two_tower_checkpoint, map_location=torch.device('cpu'))
            #knn_index.load_state_dict(checkpoint)
            #utils.load_index_file(knn_index)
            # TODO: Needs to be refactored when more datasets are available for two tower knn
            retrievers["TwoTowerKNN"] = knn_index

    for ranker_type in args.possible_rankers:
        if ranker_type == "EmbeddingRanker":
            ranker = NeuralRanker(args)
            checkpoint = torch.load(args.ranker_checkpoint, map_location=torch.device('cpu'))
            ranker.load_state_dict(checkpoint)
            rankers["EmbeddingRanker"] = ranker
        elif ranker_type == "None":
            rankers["None"] = None

    return retrievers, rankers


if __name__ == '__main__':
    args = get_args()
    retriever_models_dict, ranker_models_dict = load_models_on_start(args)
    main(args, retriever_models_dict, ranker_models_dict)
