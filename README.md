# Targeting HsDHODH for Host-Directed Therapy: Machine Learning-Guided Discovery and Structural Validation

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit-green.svg)](https://www.rdkit.org/)
[![PyCaret](https://img.shields.io/badge/AutoML-PyCaret-orange.svg)](https://pycaret.org/)

## Overview

This repository contains a complete machine learning pipeline for the identification of potent inhibitors of **HsDHODH** (*Homo sapiens* Dihydroorotate Dehydrogenase) as candidates for **Host-Directed Therapy (HDT)**. The project integrates cheminformatics, automated machine learning, and explainable AI to discover and rationalize active compounds from a curated molecular dataset.

Dihydroorotate dehydrogenase (DHODH) is a mitochondrial enzyme that catalyzes a rate-limiting step in the *de novo* pyrimidine biosynthesis pathway. By inhibiting the human isoform (HsDHODH), pathogens that depend on the host's pyrimidine pool — including *Plasmodium* spp. (malaria) and *Mycobacterium tuberculosis* (tuberculosis) — are deprived of essential metabolites, making it a validated HDT target.

## Scientific Pipeline

The project implements a QSAR (Quantitative Structure-Activity Relationship) workflow organized into five sequential stages:

```
Curated SDF Dataset (1,792 molecules)
         │
         ▼
[A] Molecular Fingerprint Generation
    ECFP (diam 2/4/6) · FCFP · MACCS Keys
         │
         ▼
[B] Baseline ML Classification (PyCaret)
    LR · RF · ET · XGB · SVM · KNN · NB
         │
         ▼
[C] Feature Engineering + Tuning
    Multicollinearity removal (90% threshold)
    SMOTE · Optuna hyperparameter search
    → 1,845 selected features
         │
         ▼
[D] Virtual Screening with Applicability Domain
    Tanimoto distance (k=5 NN, threshold=0.7)
    RF + XGB consensus predictions
         │
         ▼
[E] Explainable AI (SHAP)
    Force plots · Global importance · Fragment visualization
```

## Repository Structure

```
.
├── input/
│   └── HsDHODH-curated.sdf              # 1,792 curated molecules with activity labels
│
├── fp-gen/                               # Generated molecular fingerprints
│   ├── ECFP/
│   │   ├── diam2/                        # Morgan FP radius=1 (1024 & 2048 bits)
│   │   ├── diam4/                        # Morgan FP radius=2 — primary fingerprint
│   │   └── diam6/                        # Morgan FP radius=3 (1024 & 2048 bits)
│   ├── fcfp/                             # Feature-aware Morgan fingerprints
│   └── maccs/                            # MACCS Keys (167 bits)
│
├── output/
│   ├── selected_features_1845.txt        # Feature indices after multicollinearity filter
│   ├── shap/
│   │   ├── shap_summary_importance_RF.png
│   │   └── shap_summary_importance_XGB.png
│   └── VS_Results_AD_*.csv              # Virtual screening results with AD annotation
│
├── mlruns/                               # MLflow experiment tracking artifacts
│
├── fp_generations.py                     # Fingerprint generation module (ECFP/FCFP/MACCS)
├── fp_gen.ipynb                          # [A] Fingerprint generation pipeline
├── ML-classification-1default.ipynb     # [B] Baseline ML classification
├── ML-classification-1feature-remotion.ipynb  # [C] Feature selection, SMOTE & tuning
├── ML-classification-2prediction.ipynb  # [D] Virtual screening & applicability domain
├── XAI.ipynb                             # [E] SHAP-based model interpretability
│
├── myrdkit_env.yml                       # Conda environment (RDKit)
├── pycaret_env.yml                       # Conda environment (PyCaret + ML stack)
└── LICENSE                               # CC0 1.0 Universal
```

## Dataset

The input dataset (`HsDHODH-curated.sdf`) contains **1,792 curated small molecules** with binary activity labels:

| Class | Count | Proportion |
|-------|-------|------------|
| Active (IC₅₀ / inhibition threshold met) | 1,071 | 59.7% |
| Inactive | 721 | 40.3% |

Data curation followed standard cheminformatics practices (salt stripping, deduplication, standardization).

## Methods

### Molecular Fingerprints

Three fingerprint families were generated using RDKit via `fp_generations.py`:

| Type | Description | Variants |
|------|-------------|----------|
| **ECFP** | Extended Connectivity FP (Morgan) | Diameter 2/4/6 × 1024/2048 bits |
| **FCFP** | Feature-aware Morgan FP | Diameter 2/4/6 × 1024/2048 bits |
| **MACCS** | Structural key dictionary | Fixed 167 bits |

The **ECFP diameter-4 / 2048-bit** representation was selected as the primary input for all downstream models.

### Machine Learning

Models were trained and compared using **PyCaret 3** (automated ML framework):

- Algorithms evaluated: Logistic Regression, Random Forest, Extra Trees, XGBoost, SVM, k-NN, Naïve Bayes, Dummy
- Data split: stratified 80/20 train-test
- Cross-validation: 10-fold stratified CV
- Feature selection: multicollinearity removal at 90% Pearson threshold → **1,845 features retained**
- Class imbalance correction: **SMOTE** (Synthetic Minority Oversampling Technique)
- Hyperparameter tuning: **Optuna** (20 iterations, 4 rounds)

### Applicability Domain

Virtual screening predictions are annotated with an **Applicability Domain (AD)** score based on Tanimoto distance:

- For each query compound, the mean Tanimoto distance to the 5 nearest training molecules is computed
- Compounds with mean distance > **0.7** are flagged as outside the AD
- Predictions outside the AD are reported but should be interpreted with caution

## Results

### Classification Performance (Test Set, 1,845-feature models)

| Model | Accuracy | F1 Score | MCC | Precision | Recall |
|-------|----------|----------|-----|-----------|--------|
| Random Forest (tuned) | 87.5% | 0.8946 | 0.7401 | 0.88 | 0.91 |
| XGBoost (tuned) | 87.5% | 0.8956 | 0.7388 | 0.88 | 0.91 |
| Extra Trees (tuned) | ~87% | ~0.89 | ~0.73 | — | — |

Both RF and XGBoost achieved consistent performance, and their consensus predictions were used for virtual screening to reduce model-specific bias.

### Key Structural Features (SHAP Analysis)

SHAP (SHapley Additive exPlanations) identified **13 consensus fingerprint bits** shared between RF and XGBoost as the most predictive for HsDHODH inhibition. These bits correspond to molecular fragments including:

- Fluorinated aromatic rings
- Carboxylate-bearing substructures
- Aromatic amine motifs
- Nitrogen-containing heterocycles

SHAP summary plots are available in [output/shap/](output/shap/).

## Installation

Two Conda environments are provided — one for cheminformatics and one for ML:

```bash
# Step 1 — Chemistry / fingerprint generation
conda env create -f myrdkit_env.yml
conda activate rdkit_env

# Step 2 — Machine learning & visualization
conda env create -f pycaret_env.yml
conda activate pycaret_env
```

**Core dependencies:**

| Package | Role |
|---------|------|
| RDKit ≥ 2023 | Molecular fingerprinting, SMILES parsing |
| PyCaret 3.2 | AutoML, model comparison & serialization |
| XGBoost 2.1 | Gradient boosting classifier |
| SHAP 0.44 | Model explainability |
| MLflow 1.30 | Experiment tracking |
| Scikit-learn 1.2 | Core ML algorithms |
| Pandas / NumPy | Data handling |
| Matplotlib / Seaborn | Visualization |

## Usage

Run the notebooks in order:

```bash
jupyter notebook fp_gen.ipynb                              # [A] Generate fingerprints
jupyter notebook ML-classification-1default.ipynb         # [B] Baseline model comparison
jupyter notebook ML-classification-1feature-remotion.ipynb # [C] Feature engineering & tuning
jupyter notebook ML-classification-2prediction.ipynb       # [D] Virtual screening
jupyter notebook XAI.ipynb                                 # [E] SHAP interpretability
```

To generate fingerprints programmatically:

```python
from fp_generations import generate_ecfp, generate_fcfp, generate_maccs

# Generate ECFP4 (2048 bits) from an SDF file
df = generate_ecfp("input/HsDHODH-curated.sdf", radius=2, nBits=2048)
```

## Experiment Tracking

All runs are logged with **MLflow**. To inspect experiments:

```bash
mlflow ui
# Open http://localhost:5000
```

## License

This project is released under the **CC0 1.0 Universal** (Public Domain Dedication). You are free to copy, modify, distribute, and use the work for any purpose without asking permission. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data in your research, please cite this repository:

```
Mendonça, S.A. Targeting HsDHODH for Host-Directed Therapy:
Machine Learning-Guided Discovery and Structural Validation.
GitHub, 2025. https://github.com/sasmendonca/Targeting-HsDHODH-for-Host-Directed-Therapy-Machine-learning-Guided-Discovery-and-Structural-Valid
```

## Contact

For questions, issues, or contributions, please open a GitHub issue or reach out via the repository discussion tab.
