cd /mnt/data/lyzhuang/linear-rag

# medical
# SPACY_MODEL="en_core_sci_scibert"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET="medical"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16
# MAX_ITERATION=4
# PASSAGE_RATIO=1.5
# THRESHOLD=0.5
# TOP_K_SENTENCE=2

# musique
# SPACY_MODEL="en_core_web_trf"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET="musique"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16
# MAX_ITERATION=5
# PASSAGE_RATIO=2.0
# THRESHOLD=0.1
# TOP_K_SENTENCE=4

# 2wikimultihop
# SPACY_MODEL="en_core_web_trf"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET="2wikimultihop"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16
# MAX_ITERATION=3
# PASSAGE_RATIO=0.05
# THRESHOLD=0.4
# TOP_K_SENTENCE=1

# hotpotqa
# SPACY_MODEL="en_core_web_trf"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET="hotpotqa"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16
# MAX_ITERATION=3
# PASSAGE_RATIO=0.05
# THRESHOLD=0.4
# TOP_K_SENTENCE=1



python run.py \
    --spacy_model ${SPACY_MODEL} \
    --embedding_model ${EMBEDDING_MODEL} \
    --dataset_name ${DATASET} \
    --llm_model ${LLM_MODEL} \
    --max_workers ${MAX_WORKERS} \
    --max_iterations ${MAX_ITERATION} \
    --iteration_threshold ${THRESHOLD} \
    --passage_ratio ${PASSAGE_RATIO} \
    --top_k_sentence ${TOP_K_SENTENCE}
