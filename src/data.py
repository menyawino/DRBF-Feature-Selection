#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and preprocessing module for the Double RBF project.
Contains functions to fetch, load and preprocess gene expression data,
as well as feature selection methods.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn import svm
from scipy import stats
from geofetch import Geofetcher

import re
from pathlib import Path

def load_asd_data(dataset_id='GSE25507', force_reload=False):
    """
    Load ASD gene expression data from GEO database with sanitization.
    """
    processed_path = f'./data/processed/{dataset_id}_processed.pkl'
    
    if os.path.exists(processed_path) and not force_reload:
        print(f"Loading preprocessed data from {processed_path}")
        with open(processed_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Fetching dataset {dataset_id} from GEO...")
    
    try:
        # Initialize Geofetcher with sanitization
        geofetcher = Geofetcher(
            processed=True,
            just_metadata=False,
            destdir='./data/raw',
            skip_soft_parsing_errors=True
        )
        
        # First pass to download data
        projects = geofetcher.get_projects(dataset_id)
        
        # Clean downloaded files
        raw_dir = Path(f'./data/raw/{dataset_id}')
        for soft_file in raw_dir.glob('*.soft'):
            with open(soft_file, 'r+', encoding='utf-8', errors='replace') as f:
                content = f.read()
                # Remove control characters and invalid Unicode
                cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
                f.seek(0)
                f.write(cleaned)
                f.truncate()
        
        # Reload cleaned data
        projects = geofetcher.get_projects(dataset_id)
        project = projects[dataset_id] if isinstance(projects, dict) else projects[0]
        
        # Validate data structure
        if not hasattr(project, 'sample_table'):
            raise AttributeError("Project object missing required 'sample_table' attribute")
            
        sample_table = project.sample_table
        expr = sample_table.set_index('sample_name')
        meta = sample_table

        # Find label column with fallbacks
        label_col = next((col for col in meta.columns 
                         if any(kw in col.lower() for kw in ['diagnosis', 'group', 'condition'])), 
                        meta.columns[0])
        
        # Filter samples
        mask = meta[label_col].str.contains('autism|control', case=False, na=False, regex=True)
        if mask.sum() == 0:
            raise ValueError("No ASD/control samples found in metadata")
            
        X = expr[mask].values.astype(np.float32)
        y = np.where(meta[mask][label_col].str.contains('autism', case=False), 1, 0)
        
        # Feature selection
        gene_names = expr.columns.tolist()
        selector = VarianceThreshold(threshold=0.1)
        X = selector.fit_transform(X)
        kept_genes = selector.get_feature_names_out(input_features=gene_names)
        
        # Normalization
        X = preprocessing.StandardScaler().fit_transform(X)
        
        # Save results
        result = {'X': X, 'y': y, 'gene_names': kept_genes, 'sample_info': meta[mask]}
        with open(processed_path, 'wb') as f:
            pickle.dump(result, f)
            
        print(f"Successfully processed {X.shape[0]} samples with {X.shape[1]} features")
        return result

    except Exception as e:
        print(f"Error processing dataset {dataset_id}: {str(e)}")
        if os.path.exists(processed_path):
            os.remove(processed_path)
        raise



def select_features(X, y, gene_names, method='anova', n_features=100):
    """
    Select the most informative features from gene expression data.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Gene expression matrix (samples x genes)
    y : numpy.ndarray
        Class labels
    gene_names : list
        List of gene names corresponding to columns in X
    method : str
        Feature selection method ('anova', 'svm', or 'correlation')
    n_features : int
        Number of features to select
        
    Returns:
    --------
    tuple
        (X_selected, selected_feature_names)
    """
    print(f"Selecting top {n_features} features using {method} method...")
    
    if method == 'anova':
        # ANOVA F-value for feature selection
        selector = SelectKBest(f_classif, k=n_features)
        X_new = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [gene_names[i] for i in selected_indices]
        
    elif method == 'svm':
        # Use SVM weights for feature selection
        clf = svm.LinearSVC(C=1.0, penalty='l1', dual=False)
        clf.fit(X, y)
        coefs = clf.coef_[0]
        top_indices = np.argsort(np.abs(coefs))[-n_features:]
        X_new = X[:, top_indices]
        selected_features = [gene_names[i] for i in top_indices]
    
    elif method == 'correlation':
        # Correlation-based feature selection
        correlations = [stats.pointbiserialr(X[:, i], y)[0] for i in range(X.shape[1])]
        top_indices = np.argsort(np.abs(correlations))[-n_features:]
        X_new = X[:, top_indices]
        selected_features = [gene_names[i] for i in top_indices]
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Print top 10 selected features
    print(f"Top 10 selected features: {selected_features[:10]}")
    
    return X_new, selected_features


def load_custom_dataset(expr_file, meta_file, case_label='ASD', control_label='Control', 
                       label_column='diagnosis'):
    """
    Load a custom gene expression dataset from files.
    
    Parameters:
    -----------
    expr_file : str
        Path to expression matrix file (CSV or TSV)
    meta_file : str
        Path to metadata file (CSV or TSV)
    case_label : str
        Label for positive class in the metadata
    control_label : str
        Label for negative class in the metadata
    label_column : str
        Column name containing the class labels
        
    Returns:
    --------
    dict
        Dictionary containing the processed data
    """
    # Determine file format
    if expr_file.endswith('.csv'):
        expr_df = pd.read_csv(expr_file, index_col=0)
    else:
        expr_df = pd.read_csv(expr_file, sep='\t', index_col=0)
    
    if meta_file.endswith('.csv'):
        meta_df = pd.read_csv(meta_file, index_col=0)
    else:
        meta_df = pd.read_csv(meta_file, sep='\t', index_col=0)
    
    # Filter samples by labels
    mask = meta_df[label_column].isin([case_label, control_label])
    meta_filtered = meta_df[mask]
    
    # Get expression data for filtered samples
    common_samples = list(set(meta_filtered.index) & set(expr_df.index))
    expr_filtered = expr_df.loc[common_samples]
    meta_filtered = meta_filtered.loc[common_samples]
    
    # Create numeric labels (1 for case, 0 for control)
    y = np.where(meta_filtered[label_column] == case_label, 1, 0)
    X = expr_filtered.values
    gene_names = expr_filtered.columns.tolist()
    
    # Remove low variance features
    print(f"Original feature count: {X.shape[1]}")
    selector = VarianceThreshold(threshold=0.1)
    X = selector.fit_transform(X)
    kept_genes = np.array(gene_names)[selector.get_support()]
    print(f"Features after variance filtering: {X.shape[1]}")
    
    # Normalize data
    X = preprocessing.StandardScaler().fit_transform(X)
    
    # Create result dictionary
    result = {'X': X, 'y': y, 'gene_names': kept_genes, 'sample_info': meta_filtered}
    
    print(f"Dataset shape: {X.shape}, Positive cases: {sum(y)}, Negative cases: {len(y) - sum(y)}")
    
    return result