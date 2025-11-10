"""
Model Evaluation Script

Evaluate trained models and generate performance reports.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import MetabolomicsPreprocessor
from features import FeatureSelector
from models.baseline import BaselineModels
from models.deep import MLP, CNN1D, DeepModelTrainer
import torch
import warnings
warnings.filterwarnings('ignore')


def load_model(model_path: Path, model_type: str, input_dim: int = None, num_classes: int = 2):
    """Load trained model."""
    if model_type in ['logistic', 'random_forest', 'xgboost']:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_type == 'mlp':
        model = MLP(input_dim=input_dim, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        return model
    elif model_type == 'cnn':
        model = CNN1D(input_dim=input_dim, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_demographic_subgroups(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> dict:
    """Evaluate model performance across demographic subgroups."""
    results = {}
    
    # By sex
    if 'sex' in df.columns:
        for sex in ['M', 'F']:
            mask = df['sex'] == sex
            if mask.sum() > 0:
                results[f'sex_{sex}'] = {
                    'accuracy': accuracy_score(y[mask], y_pred[mask]),
                    'f1': f1_score(y[mask], y_pred[mask], average='weighted', zero_division=0),
                    'n_samples': mask.sum()
                }
    
    # By age range
    if 'age_range' in df.columns:
        for age_range in df['age_range'].unique():
            mask = df['age_range'] == age_range
            if mask.sum() > 0:
                results[f'age_{age_range}'] = {
                    'accuracy': accuracy_score(y[mask], y_pred[mask]),
                    'f1': f1_score(y[mask], y_pred[mask], average='weighted', zero_division=0),
                    'n_samples': mask.sum()
                }
    
    return results


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path):
    """Plot ROC curve."""
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--data', type=str, required=True, help='Path to test CSV data file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['logistic', 'random_forest', 'xgboost', 'mlp', 'cnn'],
                       help='Model to evaluate')
    parser.add_argument('--out', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--binary', action='store_true', default=True,
                       help='Binary classification')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Prepare labels
    y = (df['diagnosis_label'] != 'control').astype(int).values
    
    # Load preprocessor and feature selector
    model_dir = Path(args.model_dir)
    with open(model_dir / 'preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open(model_dir / 'feature_selector.pkl', 'rb') as f:
        feature_selector = pickle.load(f)
    
    # Preprocess
    X = preprocessor.transform(df)
    X_selected = feature_selector.transform(X)
    
    # Load model
    num_classes = len(np.unique(y))
    if args.model in ['logistic', 'random_forest', 'xgboost']:
        model_path = model_dir / f'{args.model}_model.pkl'
        model = load_model(model_path, args.model)
        
        baseline = BaselineModels()
        baseline.models[args.model] = model
        
        y_pred = baseline.predict(args.model, X_selected)
        y_proba = baseline.predict_proba(args.model, X_selected)
    else:
        model_path = model_dir / f'{args.model}_model.pth'
        model = load_model(model_path, args.model, 
                          input_dim=X_selected.shape[1], num_classes=num_classes)
        
        trainer = DeepModelTrainer(model)
        y_pred = trainer.predict(X_selected)
        y_proba = trainer.predict_proba(X_selected)
    
    # Evaluate
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
    }
    
    if y_proba is not None:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1], average='weighted')
        else:
            metrics['roc_auc'] = roc_auc_score(y, y_proba, average='weighted')
    
    # Demographic subgroup analysis
    subgroup_results = evaluate_demographic_subgroups(df, X_selected, y, y_pred, y_proba)
    metrics['subgroup_performance'] = subgroup_results
    
    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    # Save results
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(out_path / f'{args.model}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    plot_confusion_matrix(y, y_pred, out_path / f'{args.model}_confusion_matrix.png')
    if y_proba is not None:
        plot_roc_curve(y, y_proba, out_path / f'{args.model}_roc_curve.png')
    
    print(f"\nEvaluation Results for {args.model}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

