import streamlit as st


def format_retrieved_doc(doc, shortened):
    if shortened:
        length = min(1000, len(doc.raw))
    else:
        length = len(doc.raw)

    return '<br/><div style="font-family: Times New Roman; font-size: 20px;''padding-bottom:12px"><b>Score</b>: ' + \
        str(doc.score) + '<br><b>Document: ' + doc.docid + ' </b><br> ' + doc.raw[:length] + '</div>'


def show_query_results(hits, shortened, show_k=2):
    """HTML print format for the searched query"""
    for i, hit in enumerate(hits[:show_k]):
        st.write(format_retrieved_doc(hit, shortened), unsafe_allow_html=True)
