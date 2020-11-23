#!/usr/bin/python3

import streamlit as st
import torch
from utils.config import get_args


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"

#import jnius
from models.retriever.knn_retriever import KnnIndex
from models.ranker.bm25 import BM25Retriever
from models.ranker.ffn_ranker import EmbeddingRanker
import utils.utilities as utils

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
    ranker_possibilities = "none"
elif l_retriever == "KNN - Two Tower Bert":
    ranker_possibilities = ("FFN(3-layers)", "none")
l_ranker = st.sidebar.selectbox("Select ranker",
                                ranker_possibilities)

st.sidebar.subheader("Compare Options")
compare = st.sidebar.checkbox("Compare models")
if compare:
    l_retriever2 = st.sidebar.selectbox("Select another retriever",
                                        ("BM25", "KNN - Two Tower Bert"))
    l_ranker2 = st.sidebar.selectbox("Select another ranker",
                                     ("FFN(3-layers)", "none"))
st.sidebar.subheader("Other Options")

snippets = st.sidebar.checkbox("Show snippets of documents", value=True)

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
    retriever = BM25Retriever(formatter, path_to_index)
elif l_retriever == "KNN - Two Tower Bert":
    retriever = KnnIndex(args, formatter)
    state_dict = torch.load(args.two_tower_checkpoint)
    retriever.load_model_state_dict(state_dict)

ranker = None
if l_ranker == "FFN(3-layers)":
    ranker = EmbeddingRanker(args)
    checkpoint = torch.load(args.ranker_checkpoint)
    ranker.load_state_dict(checkpoint)

# Query Input for freestyle exploring
query = st.text_input("Query")

timer.reset()
hits = retriever.query(query)
retriever_time = timer.time()

timer.reset()
if ranker is not None:
    hits = utils.rerank(hits, ranker, formatter)
reranker_time = timer.time()

utils.show_query_results(hits, shortened=snippets, show_k=10)
