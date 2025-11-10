# Urine Metabolomics Cancer Screening (Prototype)

⚠️ **ETHICS WARNING**: This is a research prototype. **NOT FOR CLINICAL USE**. Real patient data must never be uploaded to public repositories. See [ETHICS.md](ETHICS.md) for details.

## Overview

This repository contains a prototype AI-based decision-support system for early screening of urological and/or kidney-related cancers using urine metabolomic profiles. The system uses machine learning and deep learning models to analyze metabolite patterns and provide preliminary risk assessments.

**⚠️ IMPORTANT**: This tool is for **research purposes only** and does **NOT** replace clinical decision-making or diagnostic procedures.

## Features

- **Synthetic Data Generator**: Creates realistic synthetic metabolomic profiles for demonstration
- **Preprocessing Pipeline**: Handles missing values, normalization, batch correction, and feature selection
- **Multiple ML Models**: 
  - Classic ML: Logistic Regression (L1), Random Forest, XGBoost
  - Deep Learning: MLP, 1D-CNN
- **Model Explainability**: SHAP-based feature importance analysis
- **Comprehensive Evaluation**: Performance metrics, demographic subgroup analysis, bias testing

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
# Generate synthetic dataset (200 samples, 500 metabolites)
python src/data/synth_generator.py --out data/synthetic --n 200 --n_metab 500
```

### 3. Run Analysis Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_EDA.ipynb - Exploratory data analysis
# 2. notebooks/02_preprocessing.ipynb - Data preprocessing
# 3. notebooks/03_baseline_models.ipynb - Model training
# 4. notebooks/04_explainability.ipynb - SHAP explanations
```

### 4. Train Models (Command Line)

```bash
# Train all models
python src/train.py --data data/synthetic/synthetic_urine_metabolomics.csv --out models --model all

# Train specific model
python src/train.py --data data/synthetic/synthetic_urine_metabolomics.csv --out models --model xgboost
```

### 5. Evaluate Models

```bash
python src/evaluate.py --data data/synthetic/synthetic_urine_metabolomics.csv --model_dir models --model xgboost --out evaluation_results
```

### 6. Generate SHAP Explanations

```bash
python src/explain.py --data data/synthetic/synthetic_urine_metabolomics.csv --model_dir models --model random_forest --out explanations
```

## Project Structure

```
urine-metabo-screening/
├── data/
│   ├── synthetic/                      # Synthetic CSV files (demo only)
│   └── README_DATA.md                  # Data handling guidelines
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory data analysis
│   ├── 02_preprocessing.ipynb         # Preprocessing pipeline
│   ├── 03_baseline_models.ipynb       # Model training & evaluation
│   └── 04_explainability.ipynb        # SHAP explanations
├── src/
│   ├── data/
│   │   └── synth_generator.py          # Synthetic data generator
│   ├── preprocessing.py                # Preprocessing pipeline
│   ├── features.py                     # Feature selection
│   ├── models/
│   │   ├── baseline.py                 # Classic ML models
│   │   └── deep.py                     # Deep learning models
│   ├── train.py                        # Training script
│   ├── evaluate.py                     # Evaluation script
│   └── explain.py                      # SHAP explanations
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── ETHICS.md                           # Ethics and legal guidelines
└── LICENSE                             # All Rights Reserved
```

## Data Format

The expected data format includes:

- **Metadata columns**: `sample_id`, `patient_id_pseudonym`, `collection_date_relative`, `age_range`, `sex`, `diagnosis_label`, `tumor_stage`, `batch_id`, `instrument_id`, etc.
- **Metabolite columns**: `metab_0001`, `metab_0002`, ..., `metab_0NNN` (LC-MS peak intensities or NMR bins)
- **QC columns**: `internal_standard_values`, `creatinine_normalization_factor`

See `data/README_DATA.md` for detailed data requirements.

## Model Performance

Models are evaluated using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion matrices
- Demographic subgroup analysis (fairness testing)
- SHAP feature importance

**Note**: Performance on synthetic data is for demonstration only. Real-world performance will vary.

## Limitations

1. **Synthetic Data**: Current implementation uses synthetic data. Real patient data requires IRB approval.
2. **Not for Clinical Use**: This is a research prototype. Do not use for clinical decision-making.
3. **Sample Size**: Models trained on small datasets may not generalize well.
4. **Batch Effects**: Real data may require more sophisticated batch correction methods.
5. **Bias**: Models should be validated across diverse demographic groups.

## Ethics & Legal

- **IRB Approval Required**: Real patient data requires Institutional Review Board approval
- **Informed Consent**: All participants must provide informed consent
- **Data Anonymization**: No PHI (Protected Health Information) in public repositories
- **Data Sharing**: Real data sharing requires Data Transfer Agreements (DTA)

See [ETHICS.md](ETHICS.md) for complete guidelines.

## Citation

If you use this code in your research, please cite appropriately and acknowledge the limitations.

## License

**All Rights Reserved** - This code is provided for research purposes only. Do not reuse without explicit permission.

## Contact

For questions about ethics, data handling, or collaboration, please contact the research team through appropriate institutional channels.

---

**⚠️ REMINDER**: This is research software. It does NOT replace clinical judgment or diagnostic procedures.

