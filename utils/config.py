import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-anserini', type=str, help="path to anserini index")
    parser.add_argument('-two_tower_checkpoint', type=str, help="path to checkpoint for two tower")
    parser.add_argument('-ranker_checkpoint', type=str, help="path to checkpoint for ranker")
    parser.add_argument('-index_file', type=str, help='path to hnswlib index')
    parser.add_argument('-index_mapping', type=str, help='path to hnswlib index')
    #parser.add_argument('-checkpoint', type=str, help='checkpoint file for two tower bert')

    parser.add_argument('-two_tower_base', type=str, default="bert-base-uncased")
    parser.add_argument('-similarity', type=str, default="ip")
    parser.add_argument('-dim_hidden', type=int, default=768)
    parser.add_argument('-max_query_len_input', type=int, default=64)
    parser.add_argument('-ranker_input', type=int, default=768)
    parser.add_argument('-ranker_hidden', type=int, default=768)
    parser.add_argument('-extra_layer', type=int, default=2500)

    args = parser.parse_args()
    return args
