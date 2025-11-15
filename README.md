# Double RBF Kernel for ASD Classification

A comprehensive machine learning project implementing customized double RBF kernels for Support Vector Machine (SVM) classification of Autism Spectrum Disorder (ASD) using gene expression data.

## Overview

This project explores the application of custom kernels for SVM classification of ASD based on gene expression profiles. The key innovation is the implementation of a "Double RBF Kernel" that combines two Radial Basis Function kernels with different gamma parameters to better capture both local and global patterns in high-dimensional gene expression data.

## Features

- Custom kernel implementation for SVM models:
  - Double RBF kernel with separate parameters for local and global patterns
  - Mixed kernel combining linear, RBF, and polynomial components
- Comprehensive data processing pipeline for gene expression data
- Multiple feature selection methods
- Model evaluation and hyperparameter tuning
- Visualization tools for model performance and interpretation
- Command line interface for easy model training and evaluation

## Project Structure

```
drbf/
├── data/
│   ├── processed/       # Processed datasets
│   └── raw/             # Raw gene expression data
├── models/              # Saved trained models
├── notebooks/           # Jupyter notebooks for exploration
├── results/             # Visualizations and result files
├── src/                 # Source code
│   ├── __init__.py      # Package initialization
│   ├── data.py          # Data loading and preprocessing functions
│   ├── models.py        # Kernel implementations and model training
│   ├── visualization.py # Plotting and visualization tools
│   └── main.py          # Main execution script
├── README.md            # Project documentation
└── requirements.txt     # Dependencies
```

## Custom Kernels

### Double RBF Kernel

The Double RBF kernel combines two RBF kernels with different gamma parameters to capture patterns at different scales:

```
K(x,y) = α * exp(-γ1||x-y||²) + (1-α) * exp(-γ2||x-y||²)
```

Where:
- γ1 is higher to capture local patterns (typically 0.1 to 1.0)
- γ2 is lower to capture global patterns (typically 0.001 to 0.01)
- α is the weight parameter controlling the balance (between 0 and 1)

### Mixed Kernel

The Mixed kernel combines linear, RBF, and polynomial kernels:

```
K(x,y) = α1 * (x·y) + α2 * [0.5 * exp(-γ1||x-y||²) + 0.5 * exp(-γ2||x-y||²)] + α3 * (x·y + 1)²
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [repository-url]
cd drbf
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Train and evaluate a model:

```bash
python src/main.py --dataset GSE25507 --feature-selection anova --n-features 500 --kernel double_rbf
```

With hyperparameter tuning:

```bash
python src/main.py --dataset GSE25507 --feature-selection anova --n-features 500 --kernel double_rbf --tune
```

### Using as a Library

```python
from src.data import load_asd_data, select_features
from src.models import train_and_evaluate, get_kernel_function
from sklearn import svm

# Load and preprocess data
data = load_asd_data()
X, y = data['X'], data['y']
gene_names = data['gene_names']

# Feature selection
X_selected, selected_features = select_features(X, y, gene_names, method='anova', n_features=500)

# Create and train a model with double RBF kernel
kernel_params = {'gamma1': 0.1, 'gamma2': 0.01, 'alpha': 0.5}
kernel_fn = get_kernel_function('double_rbf', **kernel_params)
model = svm.SVC(kernel=kernel_fn, probability=True)
model.fit(X_selected, y)

# Evaluate the model
results = train_and_evaluate(X_selected, y, kernel_type='double_rbf', **kernel_params)
print(f"Accuracy: {results['mean_accuracy']:.4f}, AUC: {results['mean_auc']:.4f}")
```

## Required Datasets

The project is designed to work with gene expression datasets from the Gene Expression Omnibus (GEO). By default, it uses the GSE25507 dataset which contains gene expression profiles from ASD and control samples.

The `geofetch` package is used to automatically download datasets from GEO. You can specify a different GEO accession using the `--dataset` parameter.

## Results Interpretation

The project generates several visualizations to help interpret the results:

1. **ROC Curves**: Comparing performance of different kernel types
2. **Feature Visualization**: t-SNE or PCA plots showing the distribution of samples
3. **Decision Boundaries**: For understanding how the model classifies samples
4. **Feature Importance**: For identifying genes most relevant to classification

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- geofetch (for GEO dataset retrieval)
