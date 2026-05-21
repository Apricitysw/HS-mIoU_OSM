#!/bin/sh

config=$1
gpus=$2
output=$3

if [ -z "$config" ]; then
    echo "No config file found! Run with: sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 1
fi

if [ -z "$gpus" ]; then
    echo "Number of gpus not specified! Run with: sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 1
fi

if [ -z "$output" ]; then
    echo "No output directory found! Run with: sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 1
fi

shift 3
opts=${@}

# --- Common Evaluation Parameters ---
# These parameters will be applied to all dataset evaluations below.
COMMON_EVAL_OPTS=" \
 EVALUATION.USE_STRUCTURED_MEASURE True \
 EVALUATION.S_ALPHA 0.5 \
 EVALUATION.HIERARCHY_GAMMA 0.3 \
 EVALUATION.HIERARCHY_BETA 0.5 \
 EVALUATION.HIERARCHY_DELTA 5 \
 EVALUATION.HIERARCHY_ALPHA 0.4 \
 EVALUATION.GLOVE_PATH HS_OSM/glove.840B.300d.txt \
"

echo "--- Starting evaluation for ADE20k-150 ---"
# ADE20k-150
python train_net.py --config-file $config \
 --num-gpus $gpus \
 --eval-only \
 OUTPUT_DIR $output/eval_ade150 \
 MODEL.WEIGHTS $output/model_final.pth \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
 DATASETS.TEST \(\"ade20k_150_test_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 EVALUATION.PRECOMPUTED_MATRIX_PATH "similarity_matrix/ade150_similarity_matrix.csv" \
 $COMMON_EVAL_OPTS \
 $opts

echo "--- Starting evaluation for Pascal Context 59 ---"
# Pascal Context 59
python train_net.py --config-file $config \
 --num-gpus $gpus \
 --eval-only \
 OUTPUT_DIR $output/eval_pc59 \
 MODEL.WEIGHTS $output/model_final.pth \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON  "datasets/pc59.json" \
 DATASETS.TEST \(\"context_59_test_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 EVALUATION.PRECOMPUTED_MATRIX_PATH "similarity_matrix/pc59_similarity_matrix.csv" \
 $COMMON_EVAL_OPTS \
 $opts

echo "--- Starting evaluation for Pascal Context 459 ---"
# Pascal Context 459
python train_net.py --config-file $config \
 --num-gpus $gpus \
 --eval-only \
 OUTPUT_DIR $output/eval_pc459 \
 MODEL.WEIGHTS $output/model_final.pth \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/pc459.json" \
 DATASETS.TEST \(\"context_459_test_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 EVALUATION.GLOVE_MAPPING_FILE "HS_OSM/word_missing_mapping/pc459_glove_mappings.json" \
 EVALUATION.WORDNET_MAPPING_FILE "HS_OSM/word_missing_mapping/pc459_wordnet_mappings.json" \
 EVALUATION.PRECOMPUTED_MATRIX_PATH "similarity_matrix/pc459_similarity_matrix.csv" \
 $COMMON_EVAL_OPTS \
 $opts

echo "--- All evaluations finished. ---"
echo "--- Aggregating results: ---"

# Aggregate and display the final metrics for each evaluation.
# The grep command searches for key metrics in the log files.
echo "ADE20k-150 Results:"
cat $output/eval_ade150/log.txt | grep -i "mIoU\|s-measure\|h_mIoU"

echo "Pascal Context 59 Results:"
cat $output/eval_pc59/log.txt | grep -i "mIoU\|s-measure\|h_mIoU"

echo "Pascal Context 459 Results:"
cat $output/eval_pc459/log.txt | grep -i "mIoU\|s-measure\|h_mIoU"

echo "--- Aggregation complete. ---"