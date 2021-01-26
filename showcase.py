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
from models.reformulator.query_reformulation import NeuralReformulator, TransformerReformulator, QueryReformulator
import utils.utilities as utils


timer = utils.Timer()
logger = logging.getLogger()


def main(args, retrievers, rankers, reformulators, formatter_dict):
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
    reformulator_possibilities = None
    if l_retriever == "BM25":
        ranker_possibilities = None
    elif l_retriever == "KNN - Two Tower Bert":
        ranker_possibilities = args.possible_rankers
        reformulator_possibilities = ["None"] + args.possible_reformulators

    if ranker_possibilities is not None:
        l_ranker = st.sidebar.selectbox("Select ranker",
                                        ranker_possibilities)
    else:
        l_ranker = None

    if reformulator_possibilities is not None:
        l_reformulator = st.sidebar.selectbox("Select Reformulator", tuple(reformulator_possibilities))
    else:
        l_reformulator = "None"
    st.sidebar.subheader("Compare Options")
    compare = st.sidebar.checkbox("Compare models")
    if compare:
        l_retriever2 = st.sidebar.selectbox("Select another retriever",
                                            ("BM25", "KNN - Two Tower Bert"))
        if l_retriever2 == "BM25":
            ranker_possibilities = tuple(["None"])
        elif l_retriever2 == "KNN - Two Tower Bert":
            ranker_possibilities = args.possible_rankers
            reformulator_possibilities = ["None"] + args.possible_reformulators
        l_ranker2 = st.sidebar.selectbox("Select another ranker", tuple(ranker_possibilities))
        l_reformulator2 = st.sidebar.selectbox("Select another reformulator", tuple(reformulator_possibilities))
        order_by_id = st.sidebar.checkbox("Order second ranker by doc id")
    else:
        l_ranker2 = None

    st.sidebar.subheader("Other Options")
    order_by_doc = st.sidebar.checkbox("Order second ranker by rank", value=True)
    snippets = st.sidebar.checkbox("Show snippets of documents", value=True)
    top_k = st.sidebar.number_input("How many documents to consider", value=100, min_value=1, max_value=1000)
    set_width(st.sidebar.slider("width", min_value=250, max_value=2000, value=1500))

    # Load selected models and data
    qrels = None
    queries = None
    index_key = None
    if l_dataset == "MSMARCO Doc":
        index_key = "msmarco_anserini_document"
        qrels, queries = utils.load_qrels(args.qrels_msmarco_test, args.queries_msmarco_test)
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

    reformulator = None
    if l_reformulator != "None":
        reformulator = reformulators[l_reformulator]

    # Query Input for freestyle exploring
    query = st.text_input("Query", value='the language of nature is math')
    test_queries = [(k, v) for k, v in queries.items()]
    test_queries.insert(0, ("-1", "not used"))
    test_qid, test_q = st.selectbox('select test query', tuple(test_queries))

    with st.beta_expander("Search for did"):
        input_did = st.text_input("Document ID", value='D2206089')
        if l_retriever == "KNN - Two Tower Bert":
            doc_raw = formatter.get_doc(input_did)
            st.write(utils.print_doc(input_did, doc_raw), unsafe_allow_html=True)
    relevant_doc_ids = None
    if test_qid != "-1":
        query = test_q
        relevant_doc_ids = qrels[test_qid]

    right_col = None
    if compare:
        st.write("Compare Rankers")
        left_col, right_col = st.beta_columns(2)

        ranker2 = None
        if l_ranker2 == "EmbeddingRanker":
            ranker2 = rankers[l_ranker2]
        retriever2 = None
        if l_retriever2 == "BM25":
            retriever2 = retrievers[index_key]
        elif l_retriever2 == "KNN - Two Tower Bert":
            retriever2 = retrievers["TwoTowerKNN"]
        reformulator2 = None
        if l_reformulator2 != "None":
            reformulator2 = reformulators[l_reformulator2]

    else:
        left_col = st.beta_columns(1)[0]

    with left_col:
        timer.reset()
        st.markdown(f"#### Retriever: {l_retriever}")
        st.markdown(f"#### Ranker: {l_ranker}")
        st.markdown(f"#### Reformulator: {l_reformulator}")
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
            st.write(f"Rerank: {reranker_time:.4f} seconds")
            if l_reformulator != "None":
                new_query = utils.reformulate(q_embedding, hits, reformulator, l_reformulator)
                q_embedding, hits, doc_embeddings = formatter.hnswlib_search_result(*retriever.query_embedded(new_query, k=top_k))
                hits = utils.rerank(q_embedding, doc_embeddings, hits, ranker)
        utils.show_query_results(hits, snippets, left_col, relevant_doc_ids)

    if right_col is not None:
        with right_col:
            st.write(f"#### Retriever: {l_retriever2}")
            st.write(f"#### Ranker: {l_ranker2}")
            st.write(f"#### Reformulator: {l_reformulator2}")
            st.markdown("____")
            if l_retriever2 == "BM25":
                hits2, _ = retriever2.query(query)
                hits2 = formatter.pyserini_search_result(hits2, query)[1]
            elif l_retriever2 == "KNN - Two Tower Bert":
                q_embedding, hits2, doc_embeddings = formatter.hnswlib_search_result(*retriever2.query(query, k=top_k))
            hits2 = list(hits2)
            retriever_time = timer.time()

            st.write(f"Retrieve: {len(hits2)} documents ({retriever_time:.4f} seconds)")

            if l_ranker2 == "EmbeddingRanker" and l_retriever2 == "KNN - Two Tower Bert":
                timer.reset()
                hits2 = utils.rerank(q_embedding, doc_embeddings, hits2, ranker2)
                reranker_time = timer.time()
                st.write(f"Rerank: {reranker_time:.4f} seconds")
                if l_reformulator2 != "None":
                    new_query = utils.reformulate(q_embedding, hits2, reformulator2, l_reformulator2)
                    q_embedding, hits2, doc_embeddings = formatter.hnswlib_search_result(
                        *retriever2.query_embedded(new_query, k=top_k))
                    hits2 = utils.rerank(q_embedding, doc_embeddings, hits2, ranker2)
                if order_by_id:
                    tmp_hits = []
                    # TODO: sort by doc ids of the other ranker result
                    # hits2 = tmp_hits
            utils.show_query_results(hits2, snippets, right_col, relevant_doc_ids)


@st.cache(allow_output_mutation=True, hash_funcs={SimpleSearcher: id, KnnIndex: id})
def load_models_on_start(args):
    logger.info("load_models_on_start cache miss")
    retrievers = {}
    rankers = {}
    reformulators = {}
    formatters = {}
    # Load models here
    for index_type in args.possible_knn_indexes:
        if index_type == "TwoTowerKNN":
            logger.info("load two tower index")
            knn_index = utils.load_knn_index(args)
            logger.info("loaded full index")
            retrievers["TwoTowerKNN"] = knn_index

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

    # Load Reformulator models
    for name in args.possible_reformulators:
        if name == 'Neural (h2500_top5)':
            checkpoint = torch.load(args.neural_reformulator_checkpoint, map_location=torch.device('cpu'))
            model = NeuralReformulator(top_k=5, embedding_size=768, hidden_size1=2500)
            model.load_state_dict(checkpoint)
            reformulators[name] = model
        elif name == 'Transformer (top10_h4_l1)':
            checkpoint = torch.load(args.transformer_h4_l1_checkpoint, map_location=torch.device('cpu'))
            model = TransformerReformulator(topk=10, nhead=4, num_encoder_layers=1)
            model.load_state_dict(checkpoint)
            reformulators[name] = model
        elif name == 'Transformer (top10_h6_l4)':
            checkpoint = torch.load(args.transformer_h6_l4_checkpoint, map_location=torch.device('cpu'))
            model = TransformerReformulator(topk=10, nhead=6, num_encoder_layers=4)
            model.load_state_dict(checkpoint)
            reformulators[name] = model
        elif name == 'Weighted Avg Top10':
            checkpoint = torch.load(args.weighted_avg_checkpoint, map_location=torch.device('cpu'))
            model = QueryReformulator(mode='weighted_avg', topk=10)
            model.layer.load_state_dict(checkpoint)
            reformulators[name] = model

    return retrievers, rankers, reformulators, formatters


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
    retriever_models_dict, ranker_models_dict, reformulator_dict, formatter_dict = load_models_on_start(args)
    main(args, retriever_models_dict, ranker_models_dict, reformulator_dict, formatter_dict)
