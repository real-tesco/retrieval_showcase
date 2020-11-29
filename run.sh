streamlit run showcase.py -- \
	-anserini ./data/indexes/ \
	-two_tower_checkpoint ./data/checkpoints/twotowerbert.bin \
	-index_file ./data/indexes/msmarco_knn_index_M_84_efc_100.bin \
	-index_mapping ./data/indexes/mapping_docid2indexid.json \
	-ranker_checkpoint ./data/checkpoints/ranker_extra_layer_2500.ckpt \
	-extra_layer 2500
