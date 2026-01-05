# Scripts Directory

This directory contains all executable code for AI text detection experiments.

## Python Scripts

- `prepare_data.py` - Data preparation (check NaN, prepare features, verify consistency)
- `elfen_extractor.py` - Extract linguistic features from text
- `main_classifier.py` - Main classification experiments (Testbeds 1-8)
- `ablation_study.py` - Feature ablation experiments
- `cumulative_ablation.py` - Cumulative feature removal experiments

## Bash Scripts

- `run_prepare_data.sh` - Runner for data preparation
- `run_elfen.sh` - Runner for feature extraction
- `run_classifier.sh` - Runner for classification experiments
- `run_ablation.sh` - Runner for ablation studies
- `run_cumulative_ablation.sh` - Runner for cumulative ablation

## Usage

Make scripts executable:
```bash
chmod +x run_*.sh
```

See main `README.md` in project root for detailed usage instructions.
