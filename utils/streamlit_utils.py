from models.retriever.knn_retriever import KnnIndex
import streamlit as st


@st.cache
def load_knn(args):
    st.write("Cache miss: load_knn(", args, ") ran")
    return KnnIndex(args)


def format_retrieved_doc(search_result, shortened):
    if shortened:
        length = min(1000, len(search_result[1]))
    else:
        length = len(search_result[1])

    return '<br/><div style="font-family: Times New Roman; font-size: 20px;''padding-bottom:12px"><b>Score</b>: ' + \
           str(search_result[2]) + '<br><b>Document: ' + search_result[0] + ' </b><br> ' + search_result[1][:length] + '</div>'


def show_query_results(hits, shortened, show_k=2):
    for i, hit in enumerate(hits):
        st.write(format_retrieved_doc(hit, shortened), unsafe_allow_html=True)
