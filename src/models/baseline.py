"""
Baseline Machine Learning Models

Classic ML models: Logistic Regression, Random Forest, XGBoost
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """Wrapper for baseline ML models."""
    
    def __init__(self):
        self.models = {}
        self.feature_importances = {}
        
    def train_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty: str = 'l1',
        C: float = 1.0,
        solver: str = 'liblinear'
    ):
        """Train logistic regression with L1/L2 regularization."""
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X, y)
        self.models['logistic'] = model
        
        # Feature importance (coefficient magnitude)
        if penalty == 'l1':
            self.feature_importances['logistic'] = np.abs(model.coef_[0])
        else:
            self.feature_importances['logistic'] = model.coef_[0] ** 2
        
        return model
    
    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        max_features: str = 'sqrt'
    ):
        """Train random forest classifier."""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        self.models['random_forest'] = model
        self.feature_importances['random_forest'] = model.feature_importances_
        return model
    
    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8
    ):
        """Train XGBoost classifier."""
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(X, y)
        self.models['xgboost'] = model
        self.feature_importances['xgboost'] = model.feature_importances_
        return model
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        return self.models[model_name].predict_proba(X)
    
    def cross_validate(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> dict:
        """Perform cross-validation."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        cv_scores = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=-1
        )
        
        return {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
    
    def get_feature_importance(self, model_name: str) -> np.ndarray:
        """Get feature importance for specified model."""
        if model_name not in self.feature_importances:
            raise ValueError(f"Model {model_name} not trained or importance not available")
        return self.feature_importances[model_name]

