"""
Metabolomics Data Preprocessing Pipeline

Handles missing values, normalization, batch correction, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MetabolomicsPreprocessor:
    """Preprocessing pipeline for metabolomics data."""
    
    def __init__(
        self,
        imputation_method: str = 'knn',
        normalization_method: str = 'log2',
        batch_correction: bool = True,
        scale_method: str = 'zscore'
    ):
        """
        Initialize preprocessor.
        
        Args:
            imputation_method: 'knn', 'half_min', or 'median'
            normalization_method: 'log2', 'vsn', or 'none'
            batch_correction: Whether to apply batch correction (simple mean centering)
            scale_method: 'zscore' or 'none'
        """
        self.imputation_method = imputation_method
        self.normalization_method = normalization_method
        self.batch_correction = batch_correction
        self.scale_method = scale_method
        
        self.scaler = None
        self.batch_means = {}
        self.imputer = None
        
    def get_metabolite_columns(self, df: pd.DataFrame) -> list:
        """Extract metabolite column names."""
        return [col for col in df.columns if col.startswith('metab_')]
    
    def impute_missing(self, X: np.ndarray, method: str = None) -> np.ndarray:
        """
        Impute missing values.
        
        Args:
            X: Feature matrix (samples x features)
            method: Imputation method (overrides self.imputation_method)
        """
        method = method or self.imputation_method
        X_imputed = X.copy()
        
        if method == 'knn':
            if self.imputer is None:
                self.imputer = KNNImputer(n_neighbors=5)
                X_imputed = self.imputer.fit_transform(X)
            else:
                X_imputed = self.imputer.transform(X)
        elif method == 'half_min':
            # Replace missing with half of minimum non-zero value per feature
            for i in range(X.shape[1]):
                non_zero_min = np.nanmin(X[X[:, i] > 0, i])
                if np.isnan(non_zero_min) or non_zero_min == 0:
                    non_zero_min = 1.0
                X_imputed[np.isnan(X[:, i]), i] = non_zero_min / 2
        elif method == 'median':
            # Replace with median per feature
            for i in range(X.shape[1]):
                median_val = np.nanmedian(X[:, i])
                if np.isnan(median_val):
                    median_val = 1.0
                X_imputed[np.isnan(X[:, i]), i] = median_val
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        return X_imputed
    
    def normalize(self, X: np.ndarray, method: str = None) -> np.ndarray:
        """
        Normalize data.
        
        Args:
            X: Feature matrix
            method: Normalization method (overrides self.normalization_method)
        """
        method = method or self.normalization_method
        X_norm = X.copy()
        
        if method == 'log2':
            # Log2 transform (add small constant to avoid log(0))
            X_norm = np.log2(X_norm + 1)
        elif method == 'vsn':
            # Variance Stabilizing Normalization (simplified)
            # In practice, use vsn package from Bioconductor
            X_norm = np.log2(X_norm + 1)
            # Additional variance stabilization could be added
        elif method == 'tic':
            # Total Ion Current normalization
            tic = X.sum(axis=1, keepdims=True)
            tic[tic == 0] = 1  # Avoid division by zero
            X_norm = X / tic * X.mean()
        elif method == 'creatinine':
            # This would require creatinine_normalization_factor column
            # For now, just log transform
            X_norm = np.log2(X_norm + 1)
        elif method == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return X_norm
    
    def correct_batch_effects(self, X: np.ndarray, batch_ids: np.ndarray) -> np.ndarray:
        """
        Simple batch effect correction (mean centering per batch).
        
        For more advanced methods (ComBat, limma), use external packages.
        
        Args:
            X: Feature matrix
            batch_ids: Batch identifiers for each sample
        """
        X_corrected = X.copy()
        unique_batches = np.unique(batch_ids)
        
        # Calculate batch means
        for batch in unique_batches:
            batch_mask = batch_ids == batch
            batch_mean = X[batch_mask, :].mean(axis=0, keepdims=True)
            global_mean = X.mean(axis=0, keepdims=True)
            
            # Center each batch to global mean
            X_corrected[batch_mask, :] = X_corrected[batch_mask, :] - batch_mean + global_mean
            
            # Store for transform
            self.batch_means[batch] = batch_mean - global_mean
        
        return X_corrected
    
    def scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Scale features.
        
        Args:
            X: Feature matrix
            fit: Whether to fit scaler (True for training, False for test)
        """
        if self.scale_method == 'zscore':
            if fit:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Call with fit=True first.")
                X_scaled = self.scaler.transform(X)
            return X_scaled
        elif self.scale_method == 'none':
            return X
        else:
            raise ValueError(f"Unknown scale method: {self.scale_method}")
    
    def fit_transform(self, df: pd.DataFrame, batch_col: str = 'batch_id') -> np.ndarray:
        """
        Fit preprocessor and transform training data.
        
        Args:
            df: DataFrame with metadata and metabolite columns
            batch_col: Column name for batch IDs
        """
        metab_cols = self.get_metabolite_columns(df)
        X = df[metab_cols].values
        
        # Step 1: Imputation
        X = self.impute_missing(X)
        
        # Step 2: Normalization
        X = self.normalize(X)
        
        # Step 3: Batch correction
        if self.batch_correction and batch_col in df.columns:
            batch_ids = df[batch_col].values
            X = self.correct_batch_effects(X, batch_ids)
        
        # Step 4: Scaling
        X = self.scale(X, fit=True)
        
        return X
    
    def transform(self, df: pd.DataFrame, batch_col: str = 'batch_id') -> np.ndarray:
        """
        Transform test data using fitted preprocessor.
        
        Args:
            df: DataFrame with metadata and metabolite columns
            batch_col: Column name for batch IDs
        """
        metab_cols = self.get_metabolite_columns(df)
        X = df[metab_cols].values
        
        # Step 1: Imputation
        X = self.impute_missing(X)
        
        # Step 2: Normalization
        X = self.normalize(X)
        
        # Step 3: Batch correction
        if self.batch_correction and batch_col in df.columns:
            batch_ids = df[batch_col].values
            # Apply stored batch corrections
            unique_batches = np.unique(batch_ids)
            for batch in unique_batches:
                batch_mask = batch_ids == batch
                if batch in self.batch_means:
                    X[batch_mask, :] = X[batch_mask, :] - self.batch_means[batch]
                else:
                    # New batch: center to global mean
                    batch_mean = X[batch_mask, :].mean(axis=0, keepdims=True)
                    global_mean = X.mean(axis=0, keepdims=True)
                    X[batch_mask, :] = X[batch_mask, :] - batch_mean + global_mean
        
        # Step 4: Scaling
        X = self.scale(X, fit=False)
        
        return X

