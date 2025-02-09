import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_folder_index', type=str, help="path to anserini indexes")
    parser.add_argument('-two_tower_checkpoint', type=str, help="path to checkpoint")
    parser.add_argument('-index_file', type=str, help='path to hnswlib index')
    parser.add_argument('-index_mapping', type=str, help='path to hnswlib index')
    parser.add_argument('-ranker_checkpoint', type=str, help='checkpoint file for ranker')
    parser.add_argument('-neural_reformulator_checkpoint', type=str, help='checkpoint file for reformulator')
    parser.add_argument('-transformer_h4_l1_checkpoint', type=str, help='checkpoint file for transformer1')
    parser.add_argument('-transformer_h6_l4_checkpoint', type=str, help='checkpoint file for transformer2')
    parser.add_argument('-transformer_h6_l6_checkpoint', type=str, help='checkpoint file for transformer3')
    parser.add_argument('-weighted_avg_checkpoint', type=str, help='checkpoint file for weighted avg reformulator')

    parser.add_argument('-qrels_msmarco_test', type=str, default="./data/2019qrels-docs.txt")
    parser.add_argument('-queries_msmarco_test', type=str, default="./data/msmarco-test2019-queries.tsv")
    parser.add_argument('-two_tower_base', type=str, default="bert-base-uncased")
    parser.add_argument('-similarity', type=str, default="ip")
    parser.add_argument('-dim_hidden', type=int, default=768)
    parser.add_argument('-max_query_len_input', type=int, default=64)
    parser.add_argument('-ranker_input', type=int, default=768)
    parser.add_argument('-ranker_hidden', type=int, default=768)
    parser.add_argument('-extra_layer', type=int, default=2500)

    args = parser.parse_args()
    args.train = False
    args.possible_bm25_indexes = ["msmarco_anserini_document", "msmarco_passaged_150_anserini",
                                  "index-robust04-20191213"]
    args.possible_knn_indexes = ["TwoTowerKNN"]
    args.possible_rankers = ["None", "EmbeddingRanker"]
    args.possible_reformulators = ['Neural (h2500_top5)', 'Transformer (top10_h4_l1)',    # 'Transformer(top10_h6_l4)',
                                   'Transformer (top10_h6_l4)', 'Weighted Avg Top10']
    return args
