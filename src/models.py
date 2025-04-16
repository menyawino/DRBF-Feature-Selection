#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Models module for the Double RBF project.
Contains implementations of custom kernels and model training functions.
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def double_rbf_kernel(X, Y=None, gamma1=0.1, gamma2=0.01, alpha=0.5):
    """
    Double RBF kernel combining two different gamma values.
    
    Parameters:
    -----------
    X : numpy.ndarray
        First data matrix (n_samples_X, n_features)
    Y : numpy.ndarray or None
        Second data matrix (n_samples_Y, n_features). If None, Y = X.
    gamma1 : float
        Parameter for first RBF kernel (higher value for local patterns)
    gamma2 : float
        Parameter for second RBF kernel (lower value for global patterns)
    alpha : float
        Weight for balancing the two kernels (0-1)
        
    Returns:
    --------
    numpy.ndarray
        Kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    
    # Calculate squared Euclidean distances
    squared_dist = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    
    # First RBF kernel (local patterns - higher gamma)
    rbf1 = np.exp(-gamma1 * squared_dist)
    
    # Second RBF kernel (global patterns - lower gamma)
    rbf2 = np.exp(-gamma2 * squared_dist)
    
    # Weighted combination
    return alpha * rbf1 + (1 - alpha) * rbf2


def mixed_kernel(X, Y=None, gamma1=0.1, gamma2=0.01, alpha1=0.4, alpha2=0.4, alpha3=0.2):
    """
    Mixed kernel combining linear, RBF and polynomial components.
    
    Parameters:
    -----------
    X : numpy.ndarray
        First data matrix
    Y : numpy.ndarray or None
        Second data matrix. If None, Y = X.
    gamma1, gamma2 : float
        Parameters for RBF kernels
    alpha1, alpha2, alpha3 : float
        Weights for linear, RBF, and polynomial components
        
    Returns:
    --------
    numpy.ndarray
        Kernel matrix
    """
    if Y is None:
        Y = X
    
    # Linear component
    linear_component = np.dot(X, Y.T)
    
    # RBF component (double RBF)
    squared_dist = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    rbf1 = np.exp(-gamma1 * squared_dist)  # Local patterns
    rbf2 = np.exp(-gamma2 * squared_dist)  # Global patterns
    rbf_component = 0.5 * rbf1 + 0.5 * rbf2
    
    # Polynomial component
    degree = 2
    coef0 = 1
    poly_component = (np.dot(X, Y.T) + coef0) ** degree
    
    # Weighted combination
    return alpha1 * linear_component + alpha2 * rbf_component + alpha3 * poly_component


def get_kernel_function(kernel_type, **params):
    """
    Factory function to get the appropriate kernel function.
    
    Parameters:
    -----------
    kernel_type : str
        Type of kernel ('double_rbf', 'mixed', 'rbf', 'linear', 'poly')
    **params : dict
        Kernel parameters
        
    Returns:
    --------
    callable or str
        Kernel function or name of built-in kernel
    """
    if kernel_type == 'double_rbf':
        gamma1 = params.get('gamma1', 0.1)
        gamma2 = params.get('gamma2', 0.01)
        alpha = params.get('alpha', 0.5)
        return lambda X, Y=None: double_rbf_kernel(X, Y, gamma1, gamma2, alpha)
        
    elif kernel_type == 'mixed':
        gamma1 = params.get('gamma1', 0.1)
        gamma2 = params.get('gamma2', 0.01)
        alpha1 = params.get('alpha1', 0.4)
        alpha2 = params.get('alpha2', 0.4)
        alpha3 = params.get('alpha3', 0.2)
        return lambda X, Y=None: mixed_kernel(X, Y, gamma1, gamma2, alpha1, alpha2, alpha3)
        
    elif kernel_type == 'rbf':
        gamma = params.get('gamma', 'scale')
        return 'rbf'  # Use sklearn's built-in RBF kernel
        
    elif kernel_type == 'linear':
        return 'linear'  # Use sklearn's built-in linear kernel
        
    elif kernel_type == 'poly':
        degree = params.get('degree', 2)
        return 'poly'  # Use sklearn's built-in polynomial kernel
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def train_and_evaluate(X, y, kernel_type='double_rbf', n_splits=5, **kernel_params):
    """
    Train and evaluate SVM model with specified kernel using cross-validation.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    kernel_type : str
        Type of kernel to use
    n_splits : int
        Number of cross-validation splits
    **kernel_params : dict
        Parameters for the kernel function
        
    Returns:
    --------
    dict
        Results dictionary with performance metrics
    """
    kernel = get_kernel_function(kernel_type, **kernel_params)
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Performance metrics
    tprs, aucs, accs = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    confusion_matrices = []
    reports = []
    
    # Cross-validation loop
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        clf = svm.SVC(kernel=kernel, probability=True, C=1.0)
        clf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(roc_auc)
        accs.append(clf.score(X_test, y_test))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        reports.append(classification_report(y_test, y_pred, output_dict=True))
    
    # Aggregate results
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accs)
    mean_cm = np.mean(confusion_matrices, axis=0)
    
    # Aggregate classification reports
    mean_report = {}
    for rep in reports:
        for cls, metrics in rep.items():
            if isinstance(metrics, dict):
                if cls not in mean_report:
                    mean_report[cls] = {k: [] for k in metrics.keys()}
                for metric, value in metrics.items():
                    mean_report[cls][metric].append(value)
    
    for cls, metrics in mean_report.items():
        for metric in metrics:
            metrics[metric] = np.mean(metrics[metric])
    
    results = {
        'kernel': kernel_type,
        'params': kernel_params,
        'mean_accuracy': mean_acc,
        'mean_auc': mean_auc,
        'tpr': mean_tpr,
        'cm': mean_cm,
        'report': mean_report
    }
    
    return results


def benchmark_kernels(X, y):
    """
    Compare different kernel configurations on the same dataset.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
        
    Returns:
    --------
    list
        List of result dictionaries for each kernel
    """
    kernel_configs = [
        {'name': 'Linear', 'type': 'linear'},
        {'name': 'RBF', 'type': 'rbf', 'params': {'gamma': 'scale'}},
        {'name': 'Double RBF (balanced)', 'type': 'double_rbf', 'params': {'gamma1': 0.1, 'gamma2': 0.01, 'alpha': 0.5}},
        {'name': 'Double RBF (local bias)', 'type': 'double_rbf', 'params': {'gamma1': 0.1, 'gamma2': 0.01, 'alpha': 0.7}},
        {'name': 'Double RBF (global bias)', 'type': 'double_rbf', 'params': {'gamma1': 0.1, 'gamma2': 0.01, 'alpha': 0.3}},
        {'name': 'Mixed', 'type': 'mixed', 'params': {'gamma1': 0.1, 'gamma2': 0.01}}
    ]
    
    results = []
    
    for config in kernel_configs:
        print(f"Evaluating {config['name']} kernel...")
        kernel_result = train_and_evaluate(X, y, kernel_type=config['type'], **config.get('params', {}))
        kernel_result['name'] = config['name']
        results.append(kernel_result)
        print(f"  Accuracy: {kernel_result['mean_accuracy']:.4f}, AUC: {kernel_result['mean_auc']:.4f}")
    
    return results


def hyperparameter_tuning(X, y, kernel_type='double_rbf'):
    """
    Tune hyperparameters for the specified kernel type.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    kernel_type : str
        Type of kernel to tune
        
    Returns:
    --------
    dict
        Best parameters and results
    """
    print(f"Tuning hyperparameters for {kernel_type} kernel...")
    
    if kernel_type == 'double_rbf':
        # Define parameter grid for double RBF kernel
        param_grid = [
            {'gamma1': [0.001, 0.01, 0.1, 1.0], 
             'gamma2': [0.0001, 0.001, 0.01, 0.1],
             'alpha': [0.3, 0.4, 0.5, 0.6, 0.7]}
        ]
        
        # Create model with fixed kernel for grid search
        results = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Try all combinations of parameters
        for g1 in param_grid[0]['gamma1']:
            for g2 in param_grid[0]['gamma2']:
                for a in param_grid[0]['alpha']:
                    # Skip invalid combinations (gamma1 should be > gamma2)
                    if g1 <= g2:
                        continue
                        
                    # Custom kernel function with these parameters
                    current_params = {'gamma1': g1, 'gamma2': g2, 'alpha': a}
                    result = train_and_evaluate(X, y, kernel_type=kernel_type, **current_params)
                    result['params'] = current_params
                    results.append(result)
                    print(f"  gamma1={g1}, gamma2={g2}, alpha={a}: AUC={result['mean_auc']:.4f}, Acc={result['mean_accuracy']:.4f}")
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['mean_auc'])
        print(f"Best parameters: {best_result['params']}")
        print(f"Best AUC: {best_result['mean_auc']:.4f}, Accuracy: {best_result['mean_accuracy']:.4f}")
        
        return best_result
    
    elif kernel_type == 'mixed':
        # Parameter grid for mixed kernel
        param_grid = [
            {'gamma1': [0.01, 0.1, 1.0],
             'gamma2': [0.001, 0.01, 0.1],
             'alpha1': [0.2, 0.4, 0.6],
             'alpha2': [0.2, 0.4, 0.6]}
        ]
        
        results = []
        
        # Try combinations
        for g1 in param_grid[0]['gamma1']:
            for g2 in param_grid[0]['gamma2']:
                for a1 in param_grid[0]['alpha1']:
                    for a2 in param_grid[0]['alpha2']:
                        # Make sure weights sum to ~1
                        if a1 + a2 > 0.95:
                            continue
                            
                        a3 = 1.0 - a1 - a2
                        current_params = {'gamma1': g1, 'gamma2': g2, 'alpha1': a1, 'alpha2': a2, 'alpha3': a3}
                        result = train_and_evaluate(X, y, kernel_type=kernel_type, **current_params)
                        result['params'] = current_params
                        results.append(result)
                        print(f"  gamma1={g1}, gamma2={g2}, alpha1={a1}, alpha2={a2}, alpha3={a3}: "
                              f"AUC={result['mean_auc']:.4f}, Acc={result['mean_accuracy']:.4f}")
        
        best_result = max(results, key=lambda x: x['mean_auc'])
        print(f"Best parameters: {best_result['params']}")
        print(f"Best AUC: {best_result['mean_auc']:.4f}, Accuracy: {best_result['mean_accuracy']:.4f}")
        
        return best_result
    
    else:
        # For standard kernels, use sklearn's built-in grid search
        from sklearn.model_selection import GridSearchCV
        
        if kernel_type == 'rbf':
            param_grid = {'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]}
        elif kernel_type == 'poly':
            param_grid = {'degree': [2, 3, 4], 'gamma': ['scale', 'auto', 0.1, 1.0]}
        elif kernel_type == 'linear':
            param_grid = {'C': [0.1, 1.0, 10.0, 100.0]}
        else:
            raise ValueError(f"Tuning not implemented for kernel type: {kernel_type}")
        
        # Use grid search
        clf = svm.SVC(kernel=kernel_type, probability=True)
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X, y)
        
        # Get best results
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best AUC: {grid_search.best_score_:.4f}")
        
        # Create a consistent result format
        best_result = train_and_evaluate(X, y, kernel_type=kernel_type, **best_params)
        best_result['params'] = best_params
        
        return best_result