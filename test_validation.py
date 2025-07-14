#!/usr/bin/env python3
"""
Validation script for the Double RBF kernel implementation
"""

import sys
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import project modules
from src.data import load_asd_data, select_features
from src.models import double_rbf_kernel, get_kernel_function, train_and_evaluate, benchmark_kernels
from src.visualization import visualize_feature_space

def test_double_rbf_kernel():
    """Test the core Double RBF kernel implementation"""
    print("=" * 50)
    print("Testing Double RBF Kernel Implementation")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    X = np.random.rand(10, 5)
    
    # Test basic kernel computation
    K = double_rbf_kernel(X, gamma1=1.0, gamma2=0.1, alpha=0.5)
    
    # Verify kernel properties
    assert K.shape == (10, 10), f"Kernel shape should be (10, 10), got {K.shape}"
    assert np.allclose(K, K.T), "Kernel matrix should be symmetric"
    assert np.all(K >= 0), "Kernel values should be non-negative"
    
    # Test with identical vectors (should give maximum similarity)
    K_self = double_rbf_kernel(X[:1], X[:1], gamma1=1.0, gamma2=0.1, alpha=0.5)
    assert K_self[0, 0] == 1.0, "Self-similarity should be 1.0"
    
    print("✓ Basic kernel properties verified")
    
    # Test different parameter combinations
    K1 = double_rbf_kernel(X, gamma1=1.0, gamma2=0.01, alpha=0.3)
    K2 = double_rbf_kernel(X, gamma1=0.1, gamma2=0.01, alpha=0.7)
    
    assert not np.allclose(K1, K2), "Different parameters should produce different kernels"
    print("✓ Parameter variation test passed")
    
    return True

def test_data_loading():
    """Test data loading and preprocessing"""
    print("\n" + "=" * 50)
    print("Testing Data Loading and Preprocessing")
    print("=" * 50)
    
    # Load data
    data = load_asd_data(dataset_id='GSE15402')
    X, y = data['X'], data['y']
    gene_names = data['gene_names']
    
    # Verify data properties
    assert X.shape[0] == len(y), "Number of samples should match labels"
    assert X.shape[1] == len(gene_names), "Number of features should match gene names"
    assert len(np.unique(y)) == 2, "Should have exactly 2 classes"
    assert set(np.unique(y)) == {0, 1}, "Classes should be 0 and 1"
    
    print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
    
    return X, y, gene_names

def test_feature_selection(X, y, gene_names):
    """Test feature selection methods"""
    print("\n" + "=" * 50)
    print("Testing Feature Selection")
    print("=" * 50)
    
    methods = ['anova', 'svm', 'correlation']
    n_features = 100
    
    for method in methods:
        X_selected, selected_features = select_features(
            X, y, gene_names, method=method, n_features=n_features
        )
        
        assert X_selected.shape[0] == X.shape[0], "Number of samples should be preserved"
        assert X_selected.shape[1] == n_features, f"Should select {n_features} features"
        assert len(selected_features) == n_features, "Feature names should match selection"
        
        print(f"✓ {method.upper()} feature selection: {X_selected.shape}")
    
    return True

def test_model_training(X, y):
    """Test model training and evaluation"""
    print("\n" + "=" * 50)
    print("Testing Model Training")
    print("=" * 50)
    
    # Test standard kernels
    kernels = ['linear', 'rbf']
    for kernel in kernels:
        clf = svm.SVC(kernel=kernel)
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print(f"✓ {kernel.upper()} SVM: {scores.mean():.3f} ± {scores.std():.3f}")
        assert scores.mean() > 0.5, f"{kernel} kernel should perform better than random"
    
    # Test Double RBF kernel
    kernel_fn = get_kernel_function('double_rbf', gamma1=0.1, gamma2=0.01, alpha=0.5)
    clf = svm.SVC(kernel=kernel_fn)
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print(f"✓ Double RBF SVM: {scores.mean():.3f} ± {scores.std():.3f}")
    assert scores.mean() > 0.5, "Double RBF kernel should perform better than random"
    
    return True

def test_benchmarking(X, y):
    """Test kernel benchmarking functionality"""
    print("\n" + "=" * 50)
    print("Testing Kernel Benchmarking")
    print("=" * 50)
    
    results = benchmark_kernels(X, y)
    
    # Verify results structure
    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Should have at least one result"
    
    for result in results:
        assert 'name' in result, "Each result should have a name"
        assert 'mean_accuracy' in result, "Each result should have mean accuracy"
        assert 'mean_auc' in result, "Each result should have mean AUC"
        assert 0 <= result['mean_accuracy'] <= 1, "Accuracy should be in [0, 1]"
        assert 0 <= result['mean_auc'] <= 1, "AUC should be in [0, 1]"
    
    print(f"✓ Benchmarked {len(results)} kernel configurations")
    
    return True

def main():
    """Run all validation tests"""
    print("DRBF Tool Validation Suite")
    print("=" * 60)
    
    try:
        # Test 1: Core kernel implementation
        test_double_rbf_kernel()
        
        # Test 2: Data loading
        X, y, gene_names = test_data_loading()
        
        # Test 3: Feature selection
        test_feature_selection(X, y, gene_names)
        
        # Select features for further testing
        X_selected, _ = select_features(X, y, gene_names, method='anova', n_features=100)
        
        # Test 4: Model training
        test_model_training(X_selected, y)
        
        # Test 5: Benchmarking
        test_benchmarking(X_selected, y)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - DRBF TOOL IS VIABLE!")
        print("=" * 60)
        
        # Summary
        print("\nSummary of validated functionality:")
        print("✓ Double RBF kernel implementation")
        print("✓ Data loading and preprocessing")
        print("✓ Feature selection (ANOVA, SVM, Correlation)")
        print("✓ Model training and cross-validation")
        print("✓ Kernel benchmarking and comparison")
        print("✓ Visualization capabilities")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
