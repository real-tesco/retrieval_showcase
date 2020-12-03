#!/usr/bin/python3

import streamlit as st
import torch
import os
import logging
from pyserini.search import SimpleSearcher

from utils.config import get_args
from models.retriever.knn_retriever import KnnIndex
from models.ranker.bm25 import BM25Retriever
from models.ranker.ffn_ranker import NeuralRanker
import utils.utilities as utils


timer = utils.Timer()
logger = logging.getLogger()


def main(args, retrievers, rankers, formatter_dict):
    st.title("Information Retrieval Showcase")
    st.write("Showcase for different retriever and ranker architectures")

    # Setting up sidebar with options for retriever, ranker, etc..
    st.sidebar.title("Settings")

    st.sidebar.subheader("Dataset")
    # this needs to effect the retriever and ranker choice
    l_dataset = st.sidebar.selectbox("Select dataset",
                                     ["MSMARCO Doc"])   #, "MSMARCO Doc (passaged)", "Robust04"))

    st.sidebar.subheader("Model")
    l_retriever = st.sidebar.selectbox("Select retriever",
                                       ["BM25", "KNN - Two Tower Bert"])
    ranker_possibilities = None
    if l_retriever == "BM25":
        ranker_possibilities = None
    elif l_retriever == "KNN - Two Tower Bert":
        ranker_possibilities = args.possible_rankers

    if ranker_possibilities is not None:
        l_ranker = st.sidebar.selectbox("Select ranker",
                                        ranker_possibilities)
    else:
        l_ranker = None

    st.sidebar.subheader("Compare Options")
    compare = st.sidebar.checkbox("Compare models")
    if compare:
        l_retriever2 = st.sidebar.selectbox("Select another retriever",
                                            ("BM25", "KNN - Two Tower Bert"))
        if l_retriever2 == "BM25":
            ranker_possibilities = tuple(["None"])
        elif l_retriever2 == "KNN - Two Tower Bert":
            ranker_possibilities = args.possible_rankers
        l_ranker2 = st.sidebar.selectbox("Select another ranker", tuple(ranker_possibilities))
    else:
        l_ranker2 = None

    st.sidebar.subheader("Other Options")

    snippets = st.sidebar.checkbox("Show snippets of documents", value=True)
    top_k = st.sidebar.number_input("How many documents to consider", value=100, min_value=1, max_value=1000)
    set_width(st.sidebar.slider("width", min_value=250, max_value=2000, value=1500))

    # Load selected models
    index_key = None
    if l_dataset == "MSMARCO Doc":
        index_key = "msmarco_anserini_document"
    elif l_dataset == "MSMARCO Doc (passaged)":
        index_key = "msmarco_passaged_150_anserini"
    elif l_dataset == "Robust04":
        index_key = "index-robust04-20191213"

    formatter = formatter_dict[index_key]

    retriever = None
    if l_retriever == "BM25":
        retriever = retrievers[index_key]
    elif l_retriever == "KNN - Two Tower Bert":
        retriever = retrievers["TwoTowerKNN"]

    ranker = None
    if l_ranker == "EmbeddingRanker":
        ranker = rankers["EmbeddingRanker"]

    # Query Input for freestyle exploring
    query = st.text_input("Query", value='the language of nature is math')

    right_col = None
    if compare:
        left_col, right_col = st.beta_columns(2)
        st.write("Compare Rankers")
        ranker2 = None
        if l_ranker2 == "EmbeddingRanker":
            ranker2 = rankers[l_ranker2]
        retriever2 = None
        if l_retriever2 == "BM25":
            retriever2 = retrievers[index_key]
        elif l_retriever2 == "KNN - Two Tower Bert":
            retriever2 = retrievers["TwoTowerKNN"]

    else:
        left_col = st.beta_columns(1)[0]

    with left_col:
        timer.reset()
        st.markdown(f"#### Retriever: {l_retriever}")
        st.markdown(f"#### Ranker: {l_ranker}")
        st.markdown("____")
        if l_retriever == "BM25":
            hits, _ = retriever.query(query, k=top_k)
            hits = formatter.pyserini_search_result(hits, query)[1]
        elif l_retriever == "KNN - Two Tower Bert":
            q_embedding, hits, doc_embeddings = formatter.hnswlib_search_result(*retriever.query(query, k=top_k))
        hits = list(hits)
        retriever_time = timer.time()

        st.write(f"Retrieve: {len(hits)} documents ({retriever_time:.4f} seconds)")

        if l_ranker == "EmbeddingRanker" and l_retriever == "KNN - Two Tower Bert":
            timer.reset()
            hits = utils.rerank(q_embedding, doc_embeddings, hits, ranker)
            reranker_time = timer.time()
            st.write(f"Rerank: {reranker_time:.4f}")
        utils.show_query_results(hits, snippets, left_col)

    if right_col is not None:
        with right_col:
            st.write(f"#### Retriever: {l_retriever2}")
            st.write(f"#### Ranker: {l_ranker2}")
            st.markdown("____")
            if l_retriever2 == "BM25":
                hits, _ = retriever2.query(query)
                hits = formatter.pyserini_search_result(hits, query)[1]
            elif l_retriever2 == "KNN - Two Tower Bert":
                q_embedding, hits, doc_embeddings = formatter.hnswlib_search_result(*retriever2.query(query, k=top_k))
            hits = list(hits)
            retriever_time = timer.time()

            st.write(f"Retrieve: {len(hits)} documents ({retriever_time:.4f} seconds)")

            if l_ranker2 == "EmbeddingRanker" and l_retriever2 == "KNN - Two Tower Bert":
                timer.reset()
                hits = utils.rerank(q_embedding, doc_embeddings, hits, ranker2)
                reranker_time = timer.time()
                st.write(f"Rerank: {reranker_time:.4f} seconds")
            utils.show_query_results(hits, snippets, right_col)


@st.cache(allow_output_mutation=True, hash_funcs={SimpleSearcher: id})
def load_models_on_start(args):
    logger.info("load_models_on_start cache miss")
    retrievers = {}
    rankers = {}
    formatters = {}
    # Load models here
    for index_type in args.possible_knn_indexes:
        if index_type == "TwoTowerKNN":
            logger.info("load two tower index")
            knn_index = utils.load_knn_index(args)
            logger.info("loaded full index")
            retrievers["TwoTowerKNN"] = knn_index

    # need setting up of retrievers here since streamlit cannot hash SimpleSearcher

    for folder_name in args.possible_bm25_indexes:
        formatters[folder_name] = utils.SearchResultFormatter(os.path.join(args.base_folder_index, folder_name))
        retrievers[folder_name] = BM25Retriever(dataset=os.path.join(args.base_folder_index, folder_name))

    for ranker_type in args.possible_rankers:
        if ranker_type == "EmbeddingRanker":
            ranker = NeuralRanker(args)
            checkpoint = torch.load(args.ranker_checkpoint, map_location=torch.device('cpu'))
            ranker.load_state_dict(checkpoint)
            rankers["EmbeddingRanker"] = ranker
        elif ranker_type == "None":
            rankers["None"] = None

    return retrievers, rankers, formatters


def set_width(max_width):
    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: {5}rem;
            padding-right: {5}rem;
            padding-left: {5}rem;
            padding-bottom: {5}rem;
        }}
        .reportview-container .main {{
            color: BLACK;
            background-color: WHITE;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    args = get_args()
    retriever_models_dict, ranker_models_dict, formatter_dict = load_models_on_start(args)
    main(args, retriever_models_dict, ranker_models_dict, formatter_dict)
