#!/usr/bin/python3

import streamlit as st
import torch
from utils.config import get_args
import utils.streamlit_utils as st_utils

# TODO load all models once

import os
import os
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"

#import jnius
from models.retriever.knn_retriever import KnnIndex
from models.ranker.bm25 import BM25Retriever
from models.ranker.ffn_ranker import EmbeddingRanker
import utils.utilities as utils


def main():
    timer = utils.Timer()
    args = get_args()


    st.title("Information Retrieval Showcase")
    st.write("Showcase for different retriever and ranker architectures")

    # Setting up sidebar with options for retriever, ranker, etc..
    st.sidebar.title("Settings")

    st.sidebar.subheader("Dataset")
    l_dataset = st.sidebar.selectbox("Select dataset",
                                     ("MSMARCO Doc", "MSMARCO Doc (passaged)", "Robust04"))

    st.sidebar.subheader("Model")
    l_retriever = st.sidebar.selectbox("Select retriever",
                                       ("BM25", "KNN - Two Tower Bert"))
    if l_retriever == "BM25":
        ranker_possibilities = None
    elif l_retriever == "KNN - Two Tower Bert":
        ranker_possibilities = ("FFN(3-layers)", "none")
    else:
        ranker_possibilities = None

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
        l_ranker2 = st.sidebar.selectbox("Select another ranker",
                                         ("FFN(3-layers)", "none"))
    st.sidebar.subheader("Other Options")

    snippets = st.sidebar.checkbox("Show snippets of documents", value=True)
    top_k = st.sidebar.number_input("How many documents to consider", value=100, min_value=1, max_value=1000)

    # Load selected models
    retriever = None
    path_to_index = None
    if l_dataset == "MSMARCO Doc":
        path_to_index = os.path.join(args.anserini, "msmarco_anserini_document")
    elif l_dataset == "MSMARCO Doc (passaged)":
        path_to_index = os.path.join(args.anserini, "msmarco_passaged_150_anserini")
    elif l_dataset == "Robust04":
        path_to_index = os.path.join(args.anserini, "index-robust04-20191213")

    formatter = utils.SearchResultFormatter(path_to_index)

    if l_retriever == "BM25":
        retriever = BM25Retriever(path_to_index)
    elif l_retriever == "KNN - Two Tower Bert":
        # extra function call to store result in streamlit cache
        retriever = st_utils.load_knn(args)
    retriever.set_formatter(formatter)

    ranker = None
    if l_ranker == "FFN(3-layers)":
        ranker = EmbeddingRanker(args)
        state_dict = torch.load(args.ranker_checkpoint, map_location=torch.device('cpu'))
        ranker.load_state_dict(state_dict)

    # Query Input for freestyle exploring
    query = st.text_input("Query", value='')

    timer.reset()
    hits = retriever.query(query)[1]
    retriever_time = timer.time()

    st.write(f"{retriever_time}s needed for current query to find {len(hits)} result")
    #hits = list(hits)
    print(hits)

    timer.reset()
    if ranker is not None:
        hits = utils.rerank(hits, ranker, formatter)
    reranker_time = timer.time()

    st_utils.show_query_results(hits, shortened=snippets, show_k=10)


if __name__ == '__main__':
    main()
