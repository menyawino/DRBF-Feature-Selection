#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Double RBF Kernels for ASD Classification
Main script to run the classification pipeline
"""

import argparse
import logging
from pathlib import Path
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.data import load_asd_data, select_features
from src.models import train_and_evaluate, benchmark_kernels, hyperparameter_tuning
from src.models import get_kernel_function, double_rbf_kernel
from src.visualization import visualize_results, visualize_feature_space, save_model
from sklearn import svm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drbf.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Double RBF Kernels for ASD Classification')
    
    parser.add_argument('--dataset', type=str, default='GSE25507',
                        help='GEO dataset accession number')
    parser.add_argument('--feature-selection', type=str, default='anova',
                        choices=['anova', 'svm', 'correlation'],
                        help='Feature selection method')
    parser.add_argument('--n-features', type=int, default=500,
                        help='Number of features to select')
    parser.add_argument('--kernel', type=str, default='double_rbf',
                        choices=['linear', 'rbf', 'poly', 'double_rbf', 'mixed'],
                        help='Kernel type')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds')
    
    return parser.parse_args()


def main():
    """Main function to run the ASD classification pipeline"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create necessary directories
    Path('./data/raw').mkdir(parents=True, exist_ok=True)
    Path('./data/processed').mkdir(parents=True, exist_ok=True)
    Path('./results').mkdir(parents=True, exist_ok=True)
    Path('./models').mkdir(parents=True, exist_ok=True)
    
    # 1. Load and preprocess data
    logger.info(f"Loading dataset {args.dataset}")
    data = load_asd_data(dataset_id=args.dataset)
    X, y = data['X'], data['y']
    gene_names = data['gene_names']
    
    # 2. Feature selection
    logger.info(f"Performing feature selection using {args.feature_selection} method")
    X_selected, selected_features = select_features(
        X, y, gene_names, method=args.feature_selection, n_features=args.n_features
    )
    logger.info(f"Selected {len(selected_features)} features")
    
    # 3. Visualize data
    logger.info("Generating t-SNE visualization of selected features")
    visualize_feature_space(X_selected, y, selected_features, method='tsne',
                           title='t-SNE Visualization of Selected Gene Features')
    
    # 4. Model training and evaluation
    if args.tune:
        # Hyperparameter tuning
        logger.info(f"Tuning hyperparameters for {args.kernel} kernel")
        best_params = hyperparameter_tuning(X_selected, y, kernel_type=args.kernel)
        kernel_params = best_params['params']
        logger.info(f"Best parameters: {kernel_params}")
    else:
        # Use default parameters
        if args.kernel == 'double_rbf':
            kernel_params = {'gamma1': 0.1, 'gamma2': 0.01, 'alpha': 0.5}
        elif args.kernel == 'mixed':
            kernel_params = {'gamma1': 0.1, 'gamma2': 0.01}
        else:
            kernel_params = {}
    
    # 5. Train final model
    logger.info(f"Training final model with {args.kernel} kernel")
    kernel_fn = get_kernel_function(args.kernel, **kernel_params)
    final_model = svm.SVC(kernel=kernel_fn, probability=True)
    final_model.fit(X_selected, y)
    
    # 6. Save model
    model_path = save_model(final_model, X_selected, y, args.kernel, kernel_params, selected_features)
    logger.info(f"Final model saved to {model_path}")
    
    # 7. Compare with other kernels (if requested)
    if args.kernel == 'double_rbf':
        logger.info("Benchmarking against other kernel types")
        kernel_results = benchmark_kernels(X_selected, y)
        visualize_results(kernel_results)
    
    logger.info("Pipeline completed successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())