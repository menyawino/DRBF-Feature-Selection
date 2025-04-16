#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module for the Double RBF project.
Contains functions to visualize and interpret model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_results(results):
    """
    Create visualizations for kernel performance comparison.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries from different kernels
    """
    # Setup plotting area
    plt.figure(figsize=(18, 12))
    
    # 1. ROC curves
    plt.subplot(2, 2, 1)
    mean_fpr = np.linspace(0, 1, 100)
    
    for res in results:
        plt.plot(mean_fpr, res['tpr'], 
                label=f"{res['name']} (AUC={res['mean_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    
    # 2. Accuracy comparison
    plt.subplot(2, 2, 2)
    names = [r['name'] for r in results]
    accuracies = [r['mean_accuracy'] for r in results]
    aucs = [r['mean_auc'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, aucs, width, label='AUC')
    
    plt.xlabel('Kernel Type')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Kernel Type')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    
    # 3. Confusion matrix for the best kernel
    best_idx = np.argmax([r['mean_auc'] for r in results])
    best_kernel = results[best_idx]
    
    plt.subplot(2, 2, 3)
    sns.heatmap(best_kernel['cm'], annot=True, fmt='.2f', cmap='Blues',
               xticklabels=['Control', 'ASD'],
               yticklabels=['Control', 'ASD'])
    plt.title(f"Confusion Matrix - {best_kernel['name']}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 4. Metrics table for best kernel
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    report = best_kernel['report']
    cell_text = []
    for cls in ['0', '1', 'macro avg']:
        if cls in report:
            row = [cls, f"{report[cls]['precision']:.3f}", 
                  f"{report[cls]['recall']:.3f}", 
                  f"{report[cls]['f1-score']:.3f}"]
            cell_text.append(row)
    
    table = plt.table(cellText=cell_text,
                      colLabels=['Class', 'Precision', 'Recall', 'F1-Score'],
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title(f"Classification Report - {best_kernel['name']}")
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    Path('./results').mkdir(parents=True, exist_ok=True)
    
    plt.savefig('./results/kernel_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_feature_space(X, y, features, method='tsne', title=None):
    """
    Visualize samples in 2D using dimensionality reduction.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    features : list
        List of feature names
    method : str
        Dimensionality reduction method ('tsne' or 'pca')
    title : str or None
        Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        method_name = 't-SNE'
    elif method == 'pca':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        method_name = 'PCA'
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    # Create scatter plot with different colors for classes
    plt.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='blue', marker='o', alpha=0.7, label='Control')
    plt.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='red', marker='x', alpha=0.7, label='ASD')
    
    plt.title(title or f"{method_name} Visualization of Gene Expression Profiles")
    plt.xlabel(f"{method_name} Component 1")
    plt.ylabel(f"{method_name} Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Create results directory if it doesn't exist
    Path('./results').mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(f'./results/{method}_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_model(clf, X, y, kernel_type, kernel_params, feature_names, file_prefix='best_model'):
    """
    Save the trained model and related metadata.
    
    Parameters:
    -----------
    clf : sklearn.svm.SVC
        Trained SVM classifier
    X : numpy.ndarray
        Feature matrix used for training
    y : numpy.ndarray
        Target vector
    kernel_type : str
        Type of kernel used
    kernel_params : dict
        Kernel parameters
    feature_names : list
        List of feature names
    file_prefix : str
        Prefix for saved model file
        
    Returns:
    --------
    str
        Path to the saved model file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create models directory if it doesn't exist
    Path('./models').mkdir(parents=True, exist_ok=True)
    
    model_path = f"./models/{file_prefix}_{timestamp}.pkl"
    
    # Compute performance metrics on the full dataset
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    metrics = {
        'accuracy': clf.score(X, y),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        'classification_report': classification_report(y, y_pred, output_dict=True)
    }
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y, y_prob)
        metrics['auc'] = auc(fpr, tpr)
    
    # Save model and metadata
    model_data = {
        'model': clf,
        'kernel_type': kernel_type,
        'kernel_params': kernel_params,
        'feature_names': feature_names,
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_path}")
    return model_path


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for linear SVM models.
    
    Parameters:
    -----------
    model : sklearn.svm.SVC
        Trained SVM model with linear kernel
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    """
    if not hasattr(model, 'coef_'):
        print("Feature importance is only available for linear kernel models.")
        return
    
    # Get absolute coefficients
    coefs = np.abs(model.coef_[0])
    
    # Get indices of top features
    top_indices = np.argsort(coefs)[-top_n:]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), coefs[top_indices])
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Absolute coefficient value')
    plt.title(f'Top {top_n} Features')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('./results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_decision_boundary(model, X, y, feature_names, pair_idx=(0, 1)):
    """
    Plot decision boundary for a pair of features.
    
    Parameters:
    -----------
    model : sklearn.svm.SVC
        Trained SVM model
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    feature_names : list
        List of feature names
    pair_idx : tuple
        Indices of the two features to plot
    """
    # Extract the two features
    x1_idx, x2_idx = pair_idx
    X_pair = X[:, [x1_idx, x2_idx]]
    
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create a simplified model with just these two features
    model_2d = model.__class__(**model.get_params())
    model_2d.fit(X_pair, y)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    
    # Use the simplified model to predict on the mesh grid
    if hasattr(model_2d, 'decision_function'):
        Z = model_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
        plt.colorbar()
    
    # Plot the data points
    scatter = plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y, edgecolors='k',
               cmap=plt.cm.Paired, s=50)
    
    plt.xlabel(feature_names[x1_idx])
    plt.ylabel(feature_names[x2_idx])
    plt.title(f'Decision boundary using features: {feature_names[x1_idx]} and {feature_names[x2_idx]}')
    plt.legend(*scatter.legend_elements(), title='Classes')
    
    # Save the plot
    plt.savefig('./results/decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()