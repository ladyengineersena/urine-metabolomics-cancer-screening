"""
Model Explainability using SHAP

Generate SHAP values and visualizations for model interpretability.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import shap
import matplotlib.pyplot as plt
from preprocessing import MetabolomicsPreprocessor
from features import FeatureSelector
from models.baseline import BaselineModels
import warnings
warnings.filterwarnings('ignore')


def explain_model_shap(
    model,
    X: np.ndarray,
    feature_names: list = None,
    model_type: str = 'tree',
    n_samples: int = 100
):
    """
    Generate SHAP explanations for a model.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: Names of features
        model_type: 'tree', 'linear', or 'deep'
        n_samples: Number of samples for SHAP (use subset for speed)
    """
    if feature_names is None:
        feature_names = [f"metab_{i:04d}" for i in range(X.shape[1])]
    
    # Use subset for explanation (SHAP can be slow on large datasets)
    if n_samples < len(X):
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        X_explain = X[sample_indices]
    else:
        X_explain = X
        sample_indices = np.arange(len(X))
    
    # Create appropriate SHAP explainer
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_explain)
        shap_values = explainer.shap_values(X_explain)
    else:
        # For deep models, use KernelExplainer (slower but more general)
        explainer = shap.KernelExplainer(model.predict_proba, X[:100])  # Background set
        shap_values = explainer.shap_values(X_explain)
    
    return explainer, shap_values, X_explain, sample_indices


def plot_shap_summary(shap_values, X: np.ndarray, feature_names: list, save_path: Path):
    """Plot SHAP summary plot."""
    plt.figure(figsize=(10, 8))
    
    # Handle multi-class case
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_waterfall(shap_values, X: np.ndarray, feature_names: list, 
                       sample_idx: int, save_path: Path):
    """Plot SHAP waterfall plot for a single sample."""
    plt.figure(figsize=(10, 6))
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=shap_values[sample_idx].sum(),
            data=X[sample_idx],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_top_features(shap_values, feature_names: list, top_k: int = 20) -> pd.DataFrame:
    """Get top K most important features based on mean absolute SHAP values."""
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
    
    top_features = pd.DataFrame({
        'feature': [feature_names[i] for i in top_indices],
        'mean_abs_shap': mean_abs_shap[top_indices]
    })
    
    return top_features


def main():
    parser = argparse.ArgumentParser(description='Generate SHAP explanations for models')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['logistic', 'random_forest', 'xgboost'],
                       help='Model to explain')
    parser.add_argument('--out', type=str, default='explanations', help='Output directory')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples for SHAP')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Load preprocessor and feature selector
    model_dir = Path(args.model_dir)
    with open(model_dir / 'preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open(model_dir / 'feature_selector.pkl', 'rb') as f:
        feature_selector = pickle.load(f)
    
    # Preprocess
    X = preprocessor.transform(df)
    X_selected = feature_selector.transform(X)
    
    # Get feature names
    metab_cols = [col for col in df.columns if col.startswith('metab_')]
    if hasattr(feature_selector, 'selected_features'):
        selected_metab_cols = [metab_cols[i] for i in range(len(metab_cols)) 
                             if feature_selector.selected_features[i]]
    else:
        selected_metab_cols = [f"feature_{i}" for i in range(X_selected.shape[1])]
    
    # Load model
    model_path = model_dir / f'{args.model}_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Determine model type for SHAP
    model_type_map = {
        'logistic': 'linear',
        'random_forest': 'tree',
        'xgboost': 'tree'
    }
    model_type = model_type_map[args.model]
    
    # Generate SHAP explanations
    print(f"Generating SHAP explanations for {args.model}...")
    explainer, shap_values, X_explain, sample_indices = explain_model_shap(
        model, X_selected, selected_metab_cols, model_type, args.n_samples
    )
    
    # Create output directory
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating SHAP visualizations...")
    plot_shap_summary(shap_values, X_explain, selected_metab_cols, 
                     out_path / f'{args.model}_shap_summary.png')
    
    # Waterfall plot for first sample
    plot_shap_waterfall(shap_values, X_explain, selected_metab_cols, 0,
                       out_path / f'{args.model}_shap_waterfall_sample0.png')
    
    # Get top features
    top_features = get_top_features(shap_values, selected_metab_cols, top_k=30)
    top_features.to_csv(out_path / f'{args.model}_top_features.csv', index=False)
    
    print(f"\nTop 10 most important features:")
    print(top_features.head(10))
    print(f"\nExplanations saved to {out_path}")


if __name__ == '__main__':
    main()

