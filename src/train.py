"""
Training Script

Main script to train models on preprocessed metabolomics data.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

from preprocessing import MetabolomicsPreprocessor
from features import FeatureSelector
from models.baseline import BaselineModels
from models.deep import MLP, CNN1D, DeepModelTrainer, MetabolomicsDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path: str) -> pd.DataFrame:
    """Load metabolomics dataset."""
    df = pd.read_csv(data_path)
    return df


def prepare_labels(df: pd.DataFrame, binary: bool = True) -> np.ndarray:
    """
    Prepare target labels.
    
    Args:
        df: DataFrame with diagnosis_label column
        binary: If True, binary classification (control vs cancer)
                If False, multi-class (control, prostate, bladder, kidney, other)
    """
    if binary:
        labels = (df['diagnosis_label'] != 'control').astype(int)
    else:
        label_map = {
            'control': 0,
            'cancer_prostate': 1,
            'cancer_bladder': 2,
            'cancer_kidney': 3,
            'cancer_other': 4
        }
        labels = df['diagnosis_label'].map(label_map).fillna(4).astype(int)
    
    return labels.values


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """Calculate evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_proba is not None:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1], average='weighted')
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, average='weighted')
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train metabolomics cancer screening models')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--out', type=str, default='models', help='Output directory for models')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['logistic', 'rf', 'xgboost', 'mlp', 'cnn', 'all'],
                       help='Model to train')
    parser.add_argument('--binary', action='store_true', default=True,
                       help='Binary classification (control vs cancer)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_data(args.data)
    print(f"Data shape: {df.shape}")
    
    # Prepare labels
    y = prepare_labels(df, binary=args.binary)
    print(f"Class distribution: {np.bincount(y)}")
    
    # Preprocessing
    print("Preprocessing data...")
    preprocessor = MetabolomicsPreprocessor(
        imputation_method='knn',
        normalization_method='log2',
        batch_correction=True,
        scale_method='zscore'
    )
    
    X = preprocessor.fit_transform(df)
    print(f"Preprocessed X shape: {X.shape}")
    
    # Feature selection
    print("Selecting features...")
    feature_selector = FeatureSelector(
        method='univariate',
        n_features=min(200, X.shape[1]),
        variance_threshold=0.01
    )
    X_selected = feature_selector.fit_transform(X, y)
    print(f"Selected features shape: {X_selected.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    # Create output directory
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessor and feature selector
    with open(out_path / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(out_path / 'feature_selector.pkl', 'wb') as f:
        pickle.dump(feature_selector, f)
    
    results = {}
    
    # Train baseline models
    if args.model in ['logistic', 'rf', 'xgboost', 'all']:
        print("\nTraining baseline models...")
        baseline = BaselineModels()
        
        if args.model in ['logistic', 'all']:
            print("Training Logistic Regression...")
            baseline.train_logistic_regression(X_train, y_train, penalty='l1', C=0.1)
            y_pred = baseline.predict('logistic', X_test)
            y_proba = baseline.predict_proba('logistic', X_test)
            results['logistic'] = evaluate_model(y_test, y_pred, y_proba)
            print(f"Logistic Regression - Accuracy: {results['logistic']['accuracy']:.4f}")
            
            with open(out_path / 'logistic_model.pkl', 'wb') as f:
                pickle.dump(baseline.models['logistic'], f)
        
        if args.model in ['rf', 'all']:
            print("Training Random Forest...")
            baseline.train_random_forest(X_train, y_train, n_estimators=100, max_depth=10)
            y_pred = baseline.predict('random_forest', X_test)
            y_proba = baseline.predict_proba('random_forest', X_test)
            results['random_forest'] = evaluate_model(y_test, y_pred, y_proba)
            print(f"Random Forest - Accuracy: {results['random_forest']['accuracy']:.4f}")
            
            with open(out_path / 'random_forest_model.pkl', 'wb') as f:
                pickle.dump(baseline.models['random_forest'], f)
        
        if args.model in ['xgboost', 'all']:
            print("Training XGBoost...")
            baseline.train_xgboost(X_train, y_train, n_estimators=100, max_depth=6)
            y_pred = baseline.predict('xgboost', X_test)
            y_proba = baseline.predict_proba('xgboost', X_test)
            results['xgboost'] = evaluate_model(y_test, y_pred, y_proba)
            print(f"XGBoost - Accuracy: {results['xgboost']['accuracy']:.4f}")
            
            with open(out_path / 'xgboost_model.pkl', 'wb') as f:
                pickle.dump(baseline.models['xgboost'], f)
    
    # Train deep learning models
    if args.model in ['mlp', 'cnn', 'all']:
        print("\nTraining deep learning models...")
        
        num_classes = len(np.unique(y))
        
        if args.model in ['mlp', 'all']:
            print("Training MLP...")
            mlp = MLP(input_dim=X_selected.shape[1], hidden_dims=[256, 128, 64], 
                     num_classes=num_classes, dropout=0.3)
            trainer = DeepModelTrainer(mlp, learning_rate=0.001)
            
            train_dataset = MetabolomicsDataset(X_train, y_train)
            val_dataset = MetabolomicsDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            trainer.train(train_loader, val_loader, epochs=50, early_stopping_patience=10)
            
            y_pred = trainer.predict(X_test)
            y_proba = trainer.predict_proba(X_test)
            results['mlp'] = evaluate_model(y_test, y_pred, y_proba)
            print(f"MLP - Accuracy: {results['mlp']['accuracy']:.4f}")
            
            torch.save(trainer.model.state_dict(), out_path / 'mlp_model.pth')
        
        if args.model in ['cnn', 'all']:
            print("Training 1D-CNN...")
            cnn = CNN1D(input_dim=X_selected.shape[1], num_classes=num_classes, dropout=0.3)
            trainer = DeepModelTrainer(cnn, learning_rate=0.001)
            
            train_dataset = MetabolomicsDataset(X_train, y_train)
            val_dataset = MetabolomicsDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            trainer.train(train_loader, val_loader, epochs=50, early_stopping_patience=10)
            
            y_pred = trainer.predict(X_test)
            y_proba = trainer.predict_proba(X_test)
            results['cnn'] = evaluate_model(y_test, y_pred, y_proba)
            print(f"CNN - Accuracy: {results['cnn']['accuracy']:.4f}")
            
            torch.save(trainer.model.state_dict(), out_path / 'cnn_model.pth')
    
    # Save results
    with open(out_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {out_path}")
    print("\nSummary:")
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, ROC-AUC={metrics.get('roc_auc', 'N/A')}")


if __name__ == '__main__':
    main()

