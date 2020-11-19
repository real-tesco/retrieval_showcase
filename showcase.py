#!/usr/bin/python3

import streamlit as st
import argparse
import torch

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"

from models.twotowerbert import TwoTowerBertIndex
from models.bm25 import BM25Retriever
import utils.utilities as utils

parser = argparse.ArgumentParser()
parser.add_argument('-anserini', type=str, help="path to anserini index")
parser.add_argument('-two_tower_checkpoint', type=str, help="path to checkpoint")
parser.add_argument('-two_tower_base', type=str, default="bert-base-uncased")
args = parser.parse_args()

st.title("Information Retrieval Showcase")
st.write("Showcase for different retriever and ranker architectures")

# Setting up sidebar with options for retriever, ranker, etc..
st.sidebar.title("Settings")

st.sidebar.subheader("Dataset")
l_dataset = st.sidebar.selectbox("Select dataset",
                                 ("MSMARCO Doc", "MSMARCO Doc (passaged)", "Robust04"))

st.sidebar.subheader("Model")
l_retriever = st.sidebar.selectbox("Select retriever",
                                   ("BM25", "Two Tower Bert"))
l_ranker = st.sidebar.selectbox("Select ranker",
                                ("FFN(3-layers)", "ip", "none"))

st.sidebar.subheader("Compare Options")
compare = st.sidebar.checkbox("Compare models")
if compare:
    l_retriever2 = st.sidebar.selectbox("Select another retriever",
                                        ("BM25", "Two Tower Bert"))
    l_ranker2 = st.sidebar.selectbox("Select another ranker",
                                     ("FFN(3-layers)", "ip", "none"))
st.sidebar.subheader("Other Options")

snippets = st.sidebar.checkbox("Show snippets of documents", value=True)

# Load selected models
retriever = None
if l_retriever == "BM25":
    if l_dataset == "MSMARCO Doc":
        path_to_index = os.path.join(args.anserini, "msmarco_anserini_document")
    elif l_dataset == "MSMARCO Doc (passaged)":
        path_to_index = os.path.join(args.anserini, "msmarco_passaged_150_anserini")
    elif l_dataset == "Robust04":
        path_to_index = os.path.join(args.anserini, "index-robust04-20191213")
    retriever = BM25Retriever(path_to_index)

elif l_retriever == "Two Tower Bert":
    retriever = TwoTowerBertIndex(args.two_tower_base)
    state_dict = torch.load(args.two_tower_checkpoint)
    retriever.load_state_dict(state_dict)

# Query Input for freestyle exploring
query = st.text_input("Query")

hits = retriever.query(query)


