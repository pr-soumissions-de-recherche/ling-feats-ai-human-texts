# AI Text Detection - Feature-based Classification

This repository contains code for detecting AI-generated text using linguistic feature-based classification with Support Vector Machines (SVM).

## Overview

The project evaluates feature-based approaches to distinguish between human-written and AI-generated text across multiple domains and model families using the MAGE benchmark dataset.

### Key Features

- **284 Linguistic Features** organized into 11 categories: surface, lexical richness, emotion, psycholinguistic, readability, dependency, entities, morphological, semantic, information, and POS
- **8 Testbeds** ranging from baseline scenarios to challenging cross-generalization settings
- **Ablation Studies** to assess individual feature group contributions
- **Cumulative Ablation** to understand progressive feature removal impact

## Repository Structure

```
.
├── scripts/                  # All Python scripts and bash runners
│   ├── check_nan_values.py
│   ├── elfen_extractor.py
│   ├── main_classifier.py
│   ├── ablation_study.py
│   ├── cumulative_ablation.py
│   ├── run_check_nan.sh
│   ├── run_elfen.sh
│   ├── run_classifier.sh
│   ├── run_ablation.sh
│   └── run_cumulative_ablation.sh
├── data/                     # Data directory
│   ├── mage/                 # MAGE dataset with features
│   ├── cmv/                  # CMV dataset with features
│   ├── raw/                  # Raw text data
│   └── feature_consistency_report.json
├── results/                  # Experimental results
│   ├── ablation/
│   └── cumulative_ablation/
└── logs/                     # Experiment logs
    ├── ablation/
    └── cumulative_ablation/
```

## Requirements

```
python>=3.8
numpy
pandas
scikit-learn
scikit-learn-intelex
polars
elfen
joblib
```

## Usage

### 1. Feature Extraction

Extract linguistic features from raw text data:

```bash
cd scripts
chmod +x run_elfen.sh
./run_elfen.sh
```

Or using Python directly:

```bash
python elfen_extractor.py --input <input_file> --output ../data/features --normalize token --combined
```

### 2. Data Preparation and Validation

Comprehensive data preparation (check for errors, prepare features, verify consistency):

```bash
chmod +x run_prepare_data.sh
./run_prepare_data.sh
```

Or run individual steps:

```bash
# Check for NaN/Inf values
python prepare_data.py --mode check --data-dir ../data/features

# Prepare features (remove zero-variance, align datasets)
python prepare_data.py --mode prepare --mage-dir ../data/mage_raw --cmv-dir ../data/cmv_raw --output-dir ../data/features

# Verify feature consistency
python prepare_data.py --mode verify --data-dir ../data/features

# Run all steps
python prepare_data.py --mode all --mage-dir ../data/mage_raw --cmv-dir ../data/cmv_raw --output-dir ../data/features
```

### 3. Run Main Classification Experiments

Run experiments on specific testbeds:

```bash
chmod +x run_classifier.sh
./run_classifier.sh
```

Or using Python directly:

```bash
python main_classifier.py --testbed 4 --feature-group combined --data-path ../data/mage
```

Available testbeds:
- **1**: Fixed-domain & Model-specific
- **11**: Fixed-domain & Model-family-specific
- **2**: Arbitrary-domains & Model-specific
- **3**: Fixed-domain & Arbitrary-models
- **4**: Arbitrary-domains & Arbitrary-models
- **5**: Unseen Models
- **6**: Unseen Domains
- **7**: Unseen-domains (all new domains) & Unseen-model
- **7.1**: Unseen-domain (each new domain) & Unseen-model
- **8**: Unseen Domain-Model Pair

### 4. Run Ablation Studies

Systematically remove feature groups to assess their impact:

```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

Or using Python directly:

```bash
python ablation_study.py --testbed 4 --feature-group combined --feature-json ../data/feature_consistency_report.json
```

### 5. Run Cumulative Ablation

Progressively remove feature groups:

```bash
chmod +x run_cumulative_ablation.sh
./run_cumulative_ablation.sh
```

Or using Python directly:

```bash
python cumulative_ablation.py --testbed 7 --ablation-order "morphological,psycholinguistic,information,dependency,surface,pos,emotion,entities,semantic,readability,lexical_richness"
```

## Configuration

All scripts support command-line arguments. Edit the bash scripts or pass arguments directly:

### Common Arguments

- `--data-path`: Path to MAGE dataset (default: `../data/mage`) # After feature extraction
- `--cmv-path`: Path to CMV dataset (default: `../data/cmv`) # After feature extraction
- `--output-dir`: Output directory for results (default: `../results`)
- `--log-dir`: Directory for log files (default: `../logs`)
- `--feature-group`: Feature group to use (default: `combined`)
- `--run-cmv`: Also run CMV testbeds (flag)

## Feature Groups

The 11 feature categories are:

1. **Surface**
2. **Lexical Richness**
3. **Emotion**
4. **Psycholinguistic**
5. **Readability**
6. **Dependency**
7. **Entities**
8. **Morphological**
9. **Semantic**
10. **Information**
11. **POS**

## Results

Results are saved in the `results/` directory with the following structure:

- Classification metrics (accuracy, AUROC, F1-macro)
- Per-class metrics (precision, recall, F1)
- Confusion matrices
- Top feature coefficients
- Global summaries in JSON format

## Notes

- All experiments use SVM with linear kernel and balanced class weighting
- Random seed is set to 42 for reproducibility
- Train/validation/test splits maintain class stratification

## Acknowledgments

Code organization and documentation assisted by Claude- Sonnet 4.5 (Anthropic).

## License

This code is provided for research purposes.
