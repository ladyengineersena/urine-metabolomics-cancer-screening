"""
Feature Selection and Dimensionality Reduction

Methods for selecting important metabolites and reducing dimensionality.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from umap import UMAP
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """Feature selection and dimensionality reduction for metabolomics."""
    
    def __init__(
        self,
        method: str = 'univariate',
        n_features: int = 100,
        variance_threshold: float = 0.01
    ):
        """
        Initialize feature selector.
        
        Args:
            method: 'univariate', 'mutual_info', 'pca', 'lda', 'umap', or 'none'
            n_features: Number of features to select (for univariate/mutual_info)
            variance_threshold: Minimum variance threshold
        """
        self.method = method
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.selector = None
        self.selected_features = None
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Fit selector and transform data.
        
        Args:
            X: Feature matrix (samples x features)
            y: Target labels (required for supervised methods)
        """
        # Step 1: Remove low variance features
        if self.variance_threshold > 0:
            var_selector = VarianceThreshold(threshold=self.variance_threshold)
            X = var_selector.fit_transform(X)
            self.var_selector = var_selector
        
        if self.method == 'none':
            return X
        
        if self.method == 'univariate':
            if y is None:
                raise ValueError("y required for univariate feature selection")
            self.selector = SelectKBest(score_func=f_classif, k=min(self.n_features, X.shape[1]))
            X_selected = self.selector.fit_transform(X, y)
            self.selected_features = self.selector.get_support()
            
        elif self.method == 'mutual_info':
            if y is None:
                raise ValueError("y required for mutual information feature selection")
            self.selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_features, X.shape[1]))
            X_selected = self.selector.fit_transform(X, y)
            self.selected_features = self.selector.get_support()
            
        elif self.method == 'pca':
            self.selector = PCA(n_components=min(self.n_features, X.shape[1]))
            X_selected = self.selector.fit_transform(X)
            
        elif self.method == 'lda':
            if y is None:
                raise ValueError("y required for LDA")
            n_components = min(len(np.unique(y)) - 1, self.n_features, X.shape[1])
            self.selector = LinearDiscriminantAnalysis(n_components=n_components)
            X_selected = self.selector.fit_transform(X, y)
            
        elif self.method == 'umap':
            self.selector = UMAP(
                n_components=min(self.n_features, X.shape[1]),
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
            X_selected = self.selector.fit_transform(X)
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
        
        return X_selected
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted selector."""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit_transform first.")
        
        # Apply variance threshold if used
        if hasattr(self, 'var_selector'):
            X = self.var_selector.transform(X)
        
        if self.method == 'none':
            return X
        
        return self.selector.transform(X)
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance scores (for univariate/mutual_info methods).
        
        Args:
            feature_names: Original feature names
        """
        if self.method not in ['univariate', 'mutual_info']:
            return None
        
        if self.selector is None:
            return None
        
        scores = self.selector.scores_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(scores))]
        
        # Apply variance threshold if used
        if hasattr(self, 'var_selector'):
            selected_mask = self.var_selector.get_support()
            scores = scores[selected_mask]
            feature_names = [f for f, m in zip(feature_names, selected_mask) if m]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'score': scores
        }).sort_values('score', ascending=False)
        
        return importance_df

