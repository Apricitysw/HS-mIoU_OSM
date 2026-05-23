# HS-mIoU and OSM: Rethinking Novel Evaluation Metrics for Open-Vocabulary Semantic Segmentation

This project provides a comprehensive suite of evaluation metrics designed specifically for the task of Open-Vocabulary Semantic Segmentation. It extends traditional evaluation methods by incorporating hierarchical semantic similarity and structured measures, offering a more nuanced and human-aligned perspective on model performance.

The evaluator is built upon Detectron2's `SemSegEvaluator`, allowing for easy integration into existing Detectron2-based projects.

## Core Evaluation Metrics

In addition to standard segmentation metrics (mIoU, fwIoU, mAcc, pAcc), this suite introduces several core metrics:

### 1. Hierarchical Similarity mIoU (HS-mIoU)

**Purpose**: To address the "all-or-nothing" penalty mechanism of standard mIoU. In an open-vocabulary context, mispredicting "dog" as "wolf" is a much smaller error than predicting it as "car". H-mIoU introduces inter-class semantic similarity to implement a more reasonable penalty system.

**How it works**:
- **Similarity Calculation**: It leverages **GloVe** word vectors for base semantic similarity and adjusts these scores using the **WordNet** hierarchy (e.g., hypernyms, hyponyms).
- **Confusion Matrix Adjustment**: Before calculating IoU, the original confusion matrix is weighted by the class similarity matrix. The penalty for a misprediction is inversely proportional to its semantic similarity to the ground-truth label.
- **Formula**: `H-mIoU` is the mIoU calculated on this adjusted confusion matrix.


### 2. Structured Similarity (S-measure)

**Purpose**: To evaluate the structural consistency between the predicted mask and the ground-truth mask, compensating for IoU's inability to capture region connectivity and object integrity.

**How it works**:
S-measure is composed of two parts:
- **Object-aware Similarity (`s_object`)**: Measures the average similarity between predicted objects and ground-truth objects.
- **Region-aware Similarity (`s_region`)**: Measures the structural similarity at the region level between the predicted mask and the ground-truth mask.

The final S-measure is a weighted sum of these two components: `S-measure = α * s_object + (1 - α) * s_region`.

### 3. "Soft" & "Hard" S-measure

This is our extension of the S-measure to adapt it for the open-vocabulary setting.

- **Hard S-measure**:
  - **Definition**: The traditional S-measure. It is calculated directly between the probability map of the **target class** and the ground-truth mask.
  - **What it evaluates**: The model's ability to classify pixels into the **exact class** while preserving structural integrity.

- **Soft S-measure**:
  - **Definition**: A more lenient version of S-measure. It first identifies the Top-K classes that are most semantically similar to the ground-truth class. It then calculates the S-measure on a "tolerant mask" that includes predictions for all these Top-K classes.
  - **What it evaluates**: The model's ability to identify a **semantically correct region** and maintain its structural integrity, even if the final class label is not perfect.

## How to Use

### 1. Configuration

In your Detectron2 configuration file (`.yaml`), add or modify the `EVALUATION` block to enable these metrics.

```yaml
EVALUATION:
  USE_STRUCTURED_MEASURE: True      # Enable S-measure and H-mIoU
  S_ALPHA: 0.5                      # Weight for s_object and s_region in S-measure
  
  # H-mIoU Parameters
  HIERARCHY_GAMMA: 0.3              # Reward factor
  HIERARCHY_BETA: 0.5               # Penalty factor
  HIERARCHY_DELTA: 5                # Reward/penalty distance threshold
  HIERARCHY_ALPHA: 0.4              # Influence weight of modifiers in compound words
  
  # Dependency File Paths
  GLOVE_PATH: "path/to/your/glove.840B.300d.txt"
  PRECOMPUTED_MATRIX_PATH: "path/to/your/similarity_matrix.csv" # Optional pre-computed similarity matrix
  GLOVE_MAPPING_FILE: "path/to/glove_mappings.json"           # Optional mapping for GloVe OOV words
  WORDNET_MAPPING_FILE: "path/to/wordnet_mappings.json"         # Optional mapping for WordNet OOV words
```

### 2. Running Evaluation

The provided `eval_all.sh` script allows for convenient evaluation across multiple datasets.

```bash
# Syntax: sh eval_all.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR]
sh eval_all.sh \
    configs/config.yaml \
    8 \
    output/my_model_evaluation
```

The script will sequentially run evaluations on the ADE20k-150, Pascal-Context-59, and Pascal-Context-459 datasets, and will aggregate key results at the end.

## Outputs

After the evaluation is complete, you will find the following files in your specified output directory (e.g., `output/my_model_evaluation/eval_ade150`):

- `log.txt`: A detailed log containing all metrics, including mIoU, h_mIoU, IoU_open, Hard-S-measure, Soft-S-measure, etc.
- `overall_results.txt`: A formatted summary of the main metrics.
- `per_class_results.txt`: Detailed metrics for each class, useful for analyzing model performance on specific categories.
- `per_image_metrics.txt`: Evaluation metrics for each individual image.

## Utility Scripts

This project also includes utility scripts to handle common Out-of-Vocabulary (OOV) issues in open-vocabulary datasets:

- `create_glove_mappings.py`: Creates mappings for compound words (e.g., "fireextinguisher") that do not exist in GloVe by splitting them into constituent parts (e.g., `["fire", "extinguisher"]`).
- `create_wordnet_mappings.py`: Creates similar mappings for words that are OOV in WordNet.