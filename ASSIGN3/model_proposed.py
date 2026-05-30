# # # =============================================================================
# # # FILE: model_proposed.py
# # # =============================================================================
# # """
# # Proposed Model for Assignment 3: Enhanced Ridge Regression
# # Improvements over Baseline B:
# # 1. Polynomial feature engineering (STOI, STOI², STOI³)
# # 2. Comprehensive hyperparameter tuning (alpha search)
# # 3. Cross-validation for robust parameter selection
# # 4. Enhanced feature scaling pipeline
# # """

# # import logging
# # import pickle
# # from pathlib import Path
# # from typing import Dict, Any
# # import numpy as np
# # from sklearn.linear_model import Ridge
# # from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# # from sklearn.pipeline import Pipeline
# # from sklearn.model_selection import GridSearchCV, KFold
# # import sys

# # logger = logging.getLogger(__name__)
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# # )

# # class ProposedModelEnhanced:
# #     """
# #     Enhanced Ridge Regression with Polynomial Features and Hyperparameter Tuning.
# #     """

# #     def __init__(self, poly_degree: int = 2, cv_folds: int = 5):
# #         self.poly_degree = poly_degree
# #         self.cv_folds = cv_folds
# #         self.model_name = f"Proposed_Enhanced_Ridge_Poly{poly_degree}"
# #         self.fitted = False
# #         self.grid_search = None
# #         self.best_params = None
# #         self.cv_results = None

# #     def fit(self, X: np.ndarray, y: np.ndarray) -> None:
# #         X = np.asarray(X)
# #         y = np.asarray(y)

# #         if len(X.shape) == 1:
# #             X = X.reshape(-1, 1)

# #         if y.max() <= 1.0:
# #             logger.info("Converting correctness from 0-1 scale to 0-100 scale")
# #             y = y * 100.0

# #         logger.info(f"Fitting {self.model_name} with {len(X)} samples")

# #         pipeline = Pipeline([
# #             ('poly', PolynomialFeatures(degree=self.poly_degree, include_bias=False)),
# #             ('scaler', StandardScaler()),
# #             ('ridge', Ridge(max_iter=10000))
# #         ])

# #         param_grid = {'ridge__alpha': np.logspace(-6, 6, 25)}
# #         kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

# #         self.grid_search = GridSearchCV(
# #             estimator=pipeline,
# #             param_grid=param_grid,
# #             cv=kfold,
# #             scoring='neg_root_mean_squared_error',
# #             n_jobs=-1,
# #             verbose=2,
# #             return_train_score=True
# #         )

# #         logger.info("Starting GridSearchCV...")
# #         self.grid_search.fit(X, y)
# #         self.best_params = self.grid_search.best_params_
# #         self.cv_results = self.grid_search.cv_results_
# #         self.fitted = True

# #         best_alpha = self.best_params['ridge__alpha']
# #         best_cv_rmse = -self.grid_search.best_score_
# #         n_features = self.grid_search.best_estimator_.named_steps['poly'].n_output_features_

# #         logger.info(f"GridSearchCV complete!")
# #         logger.info(f"Best alpha: {best_alpha:.6e}")
# #         logger.info(f"Best CV RMSE: {best_cv_rmse:.4f}")
# #         logger.info(f"Number of features after polynomial expansion: {n_features}")

# #     def predict(self, X: np.ndarray) -> np.ndarray:
# #         if not self.fitted:
# #             raise RuntimeError(f"{self.model_name}: Model must be fitted before prediction.")

# #         X = np.asarray(X)
# #         if len(X.shape) == 1:
# #             X = X.reshape(-1, 1)

# #         predictions = self.grid_search.predict(X)
# #         predictions = np.clip(predictions, 0, 100)
# #         return predictions

# #     def get_params(self) -> Dict[str, Any]:
# #         if not self.fitted:
# #             return {
# #                 'model_name': self.model_name,
# #                 'fitted': False,
# #                 'poly_degree': self.poly_degree,
# #                 'cv_folds': self.cv_folds
# #             }

# #         n_features = self.grid_search.best_estimator_.named_steps['poly'].n_output_features_
# #         return {
# #             'model_name': self.model_name,
# #             'fitted': True,
# #             'poly_degree': self.poly_degree,
# #             'cv_folds': self.cv_folds,
# #             'best_alpha': float(self.best_params['ridge__alpha']),
# #             'best_cv_rmse': float(-self.grid_search.best_score_),
# #             'n_features': n_features
# #         }

# #     def save_model(self, filepath: str) -> None:
# #         if not self.fitted:
# #             logger.warning("Saving unfitted model")

# #         Path(filepath).parent.mkdir(parents=True, exist_ok=True)
# #         with open(filepath, 'wb') as f:
# #             pickle.dump(self, f)
# #         logger.info(f"Model saved to: {filepath}")

# #     @staticmethod
# #     def load_model(filepath: str) -> 'ProposedModelEnhanced':
# #         with open(filepath, 'rb') as f:
# #             model = pickle.load(f)
# #         logger.info(f"Model loaded from: {filepath}")
# #         return model

# #     def __repr__(self) -> str:
# #         if self.fitted:
# #             return (f"{self.model_name}(fitted=True, "
# #                     f"best_alpha={self.best_params['ridge__alpha']:.6e})")
# #         else:
# #             return f"{self.model_name}(fitted=False)"


# # # Quick test if run standalone
# # if __name__ == "__main__":
# #     from sklearn.model_selection import train_test_split

# #     np.random.seed(42)
# #     n_samples = 500
# #     X = np.random.rand(n_samples)
# #     y = 50 + 80 * X + np.random.normal(0, 10, n_samples)
# #     y = np.clip(y, 0, 100)

# #     model = ProposedModelEnhanced(poly_degree=3, cv_folds=3)
# #     model.fit(X, y)
# #     preds = model.predict(X)
# #     print(f"\nSample predictions: {preds[:10]}")
# #     print(f"Model params: {model.get_params()}")







# """
# Proposed Model: Enhanced Non-Linear Ridge Regression with Feature Engineering
# ICASSP Cadenza 2026 - Lyric Intelligibility Challenge

# Key Improvements over Baseline B:
# 1. Polynomial feature expansion (degree 2-3) for non-linearity
# 2. Feature engineering (sqrt, log, exponential transformations)
# 3. Ensemble model (Ridge + Lasso + ElasticNet with weighted voting)
# 4. Advanced hyperparameter tuning with cross-validation
# 5. Robust outlier handling and scaling
# 6. Feature interaction terms

# Expected RMSE: < 25 (vs Baseline B: 35.33)
# """

# import logging
# import pickle
# from pathlib import Path
# from typing import Optional, Tuple

# import numpy as np
# from sklearn.linear_model import Ridge, Lasso, ElasticNet
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
# from sklearn.model_selection import GridSearchCV, cross_val_score
# from sklearn.pipeline import Pipeline

# logger = logging.getLogger(__name__)


# class ProposedEnhancedModel:
#     """
#     Enhanced non-linear regression model with feature engineering and ensembling.
    
#     Improvements over Baseline B:
#     - Polynomial features for capturing non-linear relationships
#     - Multiple feature transformations (sqrt, log, exp)
#     - Ensemble of Ridge, Lasso, and ElasticNet
#     - Cross-validated hyperparameter tuning
#     - Robust scaling for outlier handling
#     """
    
#     def __init__(
#         self,
#         poly_degree: int = 3,
#         use_robust_scaling: bool = True,
#         ensemble_weights: Optional[Tuple[float, float, float]] = None,
#         cv_folds: int = 5,
#         verbose: bool = True
#     ):
#         """
#         Initialize the proposed enhanced model.
        
#         Args:
#             poly_degree: Degree of polynomial features (default: 3)
#             use_robust_scaling: Use RobustScaler instead of StandardScaler
#             ensemble_weights: Weights for (Ridge, Lasso, ElasticNet) ensemble
#             cv_folds: Number of cross-validation folds
#             verbose: Enable verbose logging
#         """
#         self.poly_degree = poly_degree
#         self.use_robust_scaling = use_robust_scaling
#         self.ensemble_weights = ensemble_weights or (0.5, 0.3, 0.2)
#         self.cv_folds = cv_folds
#         self.verbose = verbose
#         self.model_name = "Proposed_Enhanced_NonLinear_Ensemble"
#         self.fitted = False
        
#         # Initialize components
#         self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
#         self.poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        
#         # Ensemble models with optimized hyperparameters
#         self.ridge_model = None
#         self.lasso_model = None
#         self.elastic_model = None
        
#         # Store training statistics
#         self.train_rmse = None
#         self.cv_scores = None
#         self.feature_names = None
#         self.n_features_original = None
#         self.n_features_engineered = None
        
#     def _engineer_features(self, X: np.ndarray) -> np.ndarray:
#         """
#         Create engineered features from STOI scores.
        
#         Features created:
#         - Original STOI
#         - STOI^2, STOI^3 (polynomial)
#         - sqrt(STOI)
#         - log(STOI + epsilon)
#         - exp(STOI) - normalized
#         - STOI * log(STOI) (interaction)
        
#         Args:
#             X: STOI scores, shape (n_samples, 1)
            
#         Returns:
#             Engineered features, shape (n_samples, n_features)
#         """
#         X = np.asarray(X).reshape(-1, 1)
#         epsilon = 1e-8
        
#         # Original feature
#         features = [X]
        
#         # Polynomial features (handles X^2, X^3, etc.)
#         poly_features = self.poly_features.fit_transform(X)
#         features.append(poly_features[:, 1:])  # Exclude bias column
        
#         # Square root transformation
#         sqrt_feature = np.sqrt(X + epsilon)
#         features.append(sqrt_feature)
        
#         # Log transformation
#         log_feature = np.log(X + epsilon)
#         features.append(log_feature)
        
#         # Exponential transformation (normalized to prevent overflow)
#         exp_feature = np.exp(X - X.mean())  # Center before exp
#         features.append(exp_feature)
        
#         # Interaction: X * log(X)
#         interaction_feature = X * np.log(X + epsilon)
#         features.append(interaction_feature)
        
#         # Reciprocal transformation: 1/(X + epsilon)
#         reciprocal_feature = 1.0 / (X + epsilon)
#         features.append(reciprocal_feature)
        
#         # Concatenate all features
#         X_engineered = np.hstack(features)
        
#         if self.verbose and not self.fitted:
#             logger.info(f"Feature engineering: {X.shape[1]} -> {X_engineered.shape[1]} features")
#             self.n_features_original = X.shape[1]
#             self.n_features_engineered = X_engineered.shape[1]
        
#         return X_engineered
    
#     def _hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> dict:
#         """
#         Perform grid search cross-validation for hyperparameter tuning.
        
#         Args:
#             X: Training features (scaled)
#             y: Training labels
            
#         Returns:
#             Dictionary of best hyperparameters for each model
#         """
#         logger.info("Starting hyperparameter tuning with GridSearchCV...")
        
#         best_params = {}
        
#         # Ridge hyperparameter grid
#         ridge_params = {
#             'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
#         }
#         ridge_grid = GridSearchCV(
#             Ridge(max_iter=10000),
#             ridge_params,
#             cv=self.cv_folds,
#             scoring='neg_mean_squared_error',
#             n_jobs=-1
#         )
#         ridge_grid.fit(X, y)
#         best_params['ridge_alpha'] = ridge_grid.best_params_['alpha']
#         ridge_cv_rmse = np.sqrt(-ridge_grid.best_score_)
#         logger.info(f"Ridge - Best alpha: {best_params['ridge_alpha']:.4f}, CV RMSE: {ridge_cv_rmse:.4f}")
        
#         # Lasso hyperparameter grid
#         lasso_params = {
#             'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
#         }
#         lasso_grid = GridSearchCV(
#             Lasso(max_iter=10000),
#             lasso_params,
#             cv=self.cv_folds,
#             scoring='neg_mean_squared_error',
#             n_jobs=-1
#         )
#         lasso_grid.fit(X, y)
#         best_params['lasso_alpha'] = lasso_grid.best_params_['alpha']
#         lasso_cv_rmse = np.sqrt(-lasso_grid.best_score_)
#         logger.info(f"Lasso - Best alpha: {best_params['lasso_alpha']:.4f}, CV RMSE: {lasso_cv_rmse:.4f}")
        
#         # ElasticNet hyperparameter grid
#         elastic_params = {
#             'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
#             'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
#         }
#         elastic_grid = GridSearchCV(
#             ElasticNet(max_iter=10000),
#             elastic_params,
#             cv=self.cv_folds,
#             scoring='neg_mean_squared_error',
#             n_jobs=-1
#         )
#         elastic_grid.fit(X, y)
#         best_params['elastic_alpha'] = elastic_grid.best_params_['alpha']
#         best_params['elastic_l1_ratio'] = elastic_grid.best_params_['l1_ratio']
#         elastic_cv_rmse = np.sqrt(-elastic_grid.best_score_)
#         logger.info(f"ElasticNet - Best alpha: {best_params['elastic_alpha']:.4f}, "
#                    f"l1_ratio: {best_params['elastic_l1_ratio']:.4f}, CV RMSE: {elastic_cv_rmse:.4f}")
        
#         # Store CV scores
#         self.cv_scores = {
#             'ridge': ridge_cv_rmse,
#             'lasso': lasso_cv_rmse,
#             'elastic': elastic_cv_rmse
#         }
        
#         return best_params
    
#     def fit(self, X: np.ndarray, y: np.ndarray) -> None:
#         """
#         Fit the enhanced ensemble model to training data.
        
#         Args:
#             X: Training features (STOI scores), shape (n_samples,)
#             y: Training labels (correctness), shape (n_samples,)
#         """
#         # Input validation
#         X = np.asarray(X).reshape(-1, 1)
#         y = np.asarray(y)
        
#         if X.shape[0] != y.shape[0]:
#             raise ValueError(f"X and y must have same length: {X.shape[0]} != {y.shape[0]}")
        
#         # Convert correctness to 0-100 scale if needed
#         if y.max() <= 1.0:
#             logger.info("Converting correctness from 0-1 scale to 0-100 scale")
#             y = y * 100.0
        
#         logger.info(f"=" * 60)
#         logger.info(f"Fitting {self.model_name}")
#         logger.info(f"=" * 60)
#         logger.info(f"Training samples: {len(X)}")
#         logger.info(f"Polynomial degree: {self.poly_degree}")
#         logger.info(f"Robust scaling: {self.use_robust_scaling}")
#         logger.info(f"Ensemble weights: {self.ensemble_weights}")
#         logger.info(f"CV folds: {self.cv_folds}")
#         logger.info(f"STOI range: [{X.min():.4f}, {X.max():.4f}]")
#         logger.info(f"Correctness range: [{y.min():.4f}, {y.max():.4f}]")
        
#         # Step 1: Feature Engineering
#         logger.info("\n--- Step 1: Feature Engineering ---")
#         X_engineered = self._engineer_features(X)
#         logger.info(f"Engineered features shape: {X_engineered.shape}")
        
#         # Step 2: Scaling
#         logger.info("\n--- Step 2: Feature Scaling ---")
#         X_scaled = self.scaler.fit_transform(X_engineered)
#         logger.info(f"Scaling complete using {'RobustScaler' if self.use_robust_scaling else 'StandardScaler'}")
        
#         # Step 3: Hyperparameter Tuning
#         logger.info("\n--- Step 3: Hyperparameter Tuning ---")
#         best_params = self._hyperparameter_tuning(X_scaled, y)
        
#         # Step 4: Train Final Ensemble Models
#         logger.info("\n--- Step 4: Training Final Ensemble ---")
        
#         # Ridge
#         self.ridge_model = Ridge(alpha=best_params['ridge_alpha'], max_iter=10000)
#         self.ridge_model.fit(X_scaled, y)
#         ridge_pred = self.ridge_model.predict(X_scaled)
#         ridge_rmse = np.sqrt(np.mean((y - ridge_pred) ** 2))
#         logger.info(f"Ridge trained - Training RMSE: {ridge_rmse:.4f}")
        
#         # Lasso
#         self.lasso_model = Lasso(alpha=best_params['lasso_alpha'], max_iter=10000)
#         self.lasso_model.fit(X_scaled, y)
#         lasso_pred = self.lasso_model.predict(X_scaled)
#         lasso_rmse = np.sqrt(np.mean((y - lasso_pred) ** 2))
#         logger.info(f"Lasso trained - Training RMSE: {lasso_rmse:.4f}")
        
#         # ElasticNet
#         self.elastic_model = ElasticNet(
#             alpha=best_params['elastic_alpha'],
#             l1_ratio=best_params['elastic_l1_ratio'],
#             max_iter=10000
#         )
#         self.elastic_model.fit(X_scaled, y)
#         elastic_pred = self.elastic_model.predict(X_scaled)
#         elastic_rmse = np.sqrt(np.mean((y - elastic_pred) ** 2))
#         logger.info(f"ElasticNet trained - Training RMSE: {elastic_rmse:.4f}")
        
#         # Ensemble prediction
#         w1, w2, w3 = self.ensemble_weights
#         ensemble_pred = (w1 * ridge_pred + w2 * lasso_pred + w3 * elastic_pred)
#         ensemble_pred = np.clip(ensemble_pred, 0, 100)
        
#         self.train_rmse = np.sqrt(np.mean((y - ensemble_pred) ** 2))
        
#         self.fitted = True
        
#         logger.info(f"\n{'=' * 60}")
#         logger.info(f"ENSEMBLE TRAINING RMSE: {self.train_rmse:.4f}")
#         logger.info(f"Improvement over Baseline B (35.33): {35.33 - self.train_rmse:.4f}")
#         logger.info(f"Relative improvement: {((35.33 - self.train_rmse) / 35.33 * 100):.2f}%")
#         logger.info(f"{'=' * 60}")
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Predict correctness values from STOI scores using ensemble.
        
#         Args:
#             X: STOI scores, shape (n_samples,)
            
#         Returns:
#             Predicted correctness values (0-100), shape (n_samples,)
#         """
#         if not self.fitted:
#             raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
#         X = np.asarray(X).reshape(-1, 1)
        
#         # Feature engineering
#         X_engineered = self._engineer_features(X)
        
#         # Scaling
#         X_scaled = self.scaler.transform(X_engineered)
        
#         # Ensemble prediction
#         w1, w2, w3 = self.ensemble_weights
#         ridge_pred = self.ridge_model.predict(X_scaled)
#         lasso_pred = self.lasso_model.predict(X_scaled)
#         elastic_pred = self.elastic_model.predict(X_scaled)
        
#         predictions = (w1 * ridge_pred + w2 * lasso_pred + w3 * elastic_pred)
        
#         # Clip to valid range
#         predictions = np.clip(predictions, 0, 100)
        
#         return predictions
    
#     def get_params(self) -> dict:
#         """Get model parameters and statistics."""
#         if not self.fitted:
#             return {
#                 'model_name': self.model_name,
#                 'fitted': False,
#                 'poly_degree': self.poly_degree,
#                 'use_robust_scaling': self.use_robust_scaling,
#                 'ensemble_weights': self.ensemble_weights
#             }
        
#         return {
#             'model_name': self.model_name,
#             'fitted': True,
#             'poly_degree': self.poly_degree,
#             'use_robust_scaling': self.use_robust_scaling,
#             'ensemble_weights': self.ensemble_weights,
#             'cv_folds': self.cv_folds,
#             'train_rmse': self.train_rmse,
#             'cv_scores': self.cv_scores,
#             'n_features_original': self.n_features_original,
#             'n_features_engineered': self.n_features_engineered,
#             'ridge_alpha': self.ridge_model.alpha if self.ridge_model else None,
#             'lasso_alpha': self.lasso_model.alpha if self.lasso_model else None,
#             'elastic_alpha': self.elastic_model.alpha if self.elastic_model else None,
#             'elastic_l1_ratio': self.elastic_model.l1_ratio if self.elastic_model else None
#         }
    
#     def save_model(self, filepath: str) -> None:
#         """Save the fitted model to a pickle file."""
#         if not self.fitted:
#             logger.warning("Saving unfitted model")
        
#         Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
#         with open(filepath, 'wb') as f:
#             pickle.dump(self, f)
        
#         logger.info(f"Model saved to: {filepath}")
    
#     @staticmethod
#     def load_model(filepath: str) -> 'ProposedEnhancedModel':
#         """Load a fitted model from a pickle file."""
#         with open(filepath, 'rb') as f:
#             model = pickle.load(f)
        
#         logger.info(f"Model loaded from: {filepath}")
#         return model
    
#     def _repr_(self) -> str:
#         if self.fitted:
#             return (f"{self.model_name}(fitted=True, poly_degree={self.poly_degree}, "
#                    f"train_rmse={self.train_rmse:.4f})")
#         return f"{self.model_name}(fitted=False)"


# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     # Generate test data
#     np.random.seed(42)
#     n_samples = 1000
    
#     X_train = np.random.beta(2, 2, n_samples)
#     # Non-linear relationship: correctness = 30 + 50*X + 30*X^2 - 20*X^3 + noise
#     y_train = 30 + 50*X_train + 30*X_train*2 - 20*X_train*3 + np.random.normal(0, 8, n_samples)
#     y_train = np.clip(y_train, 0, 100)
    
#     print("\n" + "=" * 60)
#     print("Testing Proposed Enhanced Model")
#     print("=" * 60)
    
#     # Train model
#     model = ProposedEnhancedModel(
#         poly_degree=3,
#         use_robust_scaling=True,
#         ensemble_weights=(0.5, 0.3, 0.2),
#         cv_folds=5,
#         verbose=True
#     )
#     model.fit(X_train, y_train)
    
#     # Test predictions
#     X_test = np.linspace(0, 1, 10)
#     y_pred = model.predict(X_test)
    
#     print("\n=== Sample Predictions ===")
#     print("STOI Score -> Predicted Correctness")
#     for i in range(len(X_test)):
#         print(f"{X_test[i]:.2f} -> {y_pred[i]:.2f}")
    
#     # Show parameters
#     print("\n=== Model Parameters ===")
#     params = model.get_params()
#     for key, value in params.items():
#         print(f"{key}: {value}")
    
#     # Save and load test
#     model.save_model('models/proposed_model.pkl')
#     loaded_model = ProposedEnhancedModel.load_model('models/proposed_model.pkl')
#     print(f"\n✓ Model saved and loaded successfully!")
#     print(f"Loaded model: {loaded_model}")













# File: model_proposed.py

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class ProposedEnhancedModel:
    """
    Enhanced non-linear regression model with feature engineering and ensembling.
    
    Improvements over Baseline B:
    - Polynomial features for capturing non-linear relationships
    - Extensive feature engineering (sqrt, log, exp, etc.)
    - Ensemble of Ridge, Lasso, and ElasticNet
    - Cross-validated hyperparameter tuning
    - Robust scaling for outlier handling
    """
    
    def __init__(
        self,
        poly_degree: int = 2,
        use_robust_scaling: bool = True,
        ensemble_weights: Optional[Tuple[float, float, float]] = None,
        cv_folds: int = 10,  # Using 10 folds for a robust paper result
        verbose: bool = True
    ):
        """
        Initialize the proposed enhanced model.
        
        Args:
            poly_degree: Degree of polynomial features (default: 3)
            use_robust_scaling: Use RobustScaler instead of StandardScaler
            ensemble_weights: Weights for (Ridge, Lasso, ElasticNet) ensemble
            cv_folds: Number of cross-validation folds
            verbose: Enable verbose logging
        """
        self.poly_degree = poly_degree
        self.use_robust_scaling = use_robust_scaling
        self.ensemble_weights = ensemble_weights or (0.5, 0.3, 0.2)
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.model_name = "Proposed_Enhanced_NonLinear_Ensemble"
        self.fitted = False
        
        # Initialize components
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        
        # Ensemble models will be created during fit
        self.ridge_model = None
        self.lasso_model = None
        self.elastic_model = None
        
        # Store training statistics
        self.train_rmse = None
        self.cv_scores = None
        self.n_features_engineered = None
        
    def _engineer_features(self, X: np.ndarray) -> np.ndarray:
        """ Create polynomial and interaction features from the input features.
        (e.g., X1, X2, X3, X1^2, X1*X2, X1*X3, etc.)
        """
        X = np.asarray(X)

        # PolynomialFeatures is now our *entire* feature engineering.
        # It will automatically create X1^2, X2^2, X3^2, X1*X2, X1*X3, X2*X3, etc.
        # This is the core of our "Proposed Method".
        X_engineered = self.poly_features.fit_transform(X)

        self.n_features_engineered = X_engineered.shape[1]
        if self.verbose and not self.fitted:
            logger.info(f"Feature engineering: {X.shape[1]} original features -> {self.n_features_engineered} polynomial features (degree={self.poly_degree})")

        return X_engineered
    
    def _hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Perform grid search cross-validation for hyperparameter tuning.
        """
        if self.verbose:
            logger.info("Starting hyperparameter tuning with GridSearchCV...")
        
        best_params = {}
        
        # Ridge hyperparameter grid - EXPANDED RANGE
        ridge_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        }
        # --- CRITICAL FIX: Increased max_iter to prevent ConvergenceWarning ---
        ridge_grid = GridSearchCV(
            Ridge(max_iter=100000), 
            ridge_params,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        ridge_grid.fit(X, y)
        best_params['ridge_alpha'] = ridge_grid.best_params_['alpha']
        
        # Lasso hyperparameter grid - EXPANDED RANGE
        lasso_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        }
        # --- CRITICAL FIX: Increased max_iter to prevent ConvergenceWarning ---
        lasso_grid = GridSearchCV(
            Lasso(max_iter=100000), 
            lasso_params,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        lasso_grid.fit(X, y)
        best_params['lasso_alpha'] = lasso_grid.best_params_['alpha']
        
        # ElasticNet hyperparameter grid - EXPANDED RANGE
        elastic_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        # --- CRITICAL FIX: Increased max_iter to prevent ConvergenceWarning ---
        elastic_grid = GridSearchCV(
            ElasticNet(max_iter=100000),
            elastic_params,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        elastic_grid.fit(X, y)
        best_params['elastic_alpha'] = elastic_grid.best_params_['alpha']
        best_params['elastic_l1_ratio'] = elastic_grid.best_params_['l1_ratio']

        if self.verbose:
            logger.info(f"Ridge - Best alpha: {best_params['ridge_alpha']:.4f}")
            logger.info(f"Lasso - Best alpha: {best_params['lasso_alpha']:.4f}")
            logger.info(f"ElasticNet - Best alpha: {best_params['elastic_alpha']:.4f}, l1_ratio: {best_params['elastic_l1_ratio']:.4f}")
            
        return best_params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the enhanced ensemble model to training data.
        """
        y = np.asarray(y)
        
        if y.max() <= 1.0:
            if self.verbose: logger.info("Converting correctness from 0-1 scale to 0-100 scale")
            y = y * 100.0
        
        if self.verbose:
            logger.info(f"=" * 60)
            logger.info(f"Fitting {self.model_name}")
            logger.info(f"Training samples: {len(X)}")
            logger.info(f"Polynomial degree: {self.poly_degree}")
            logger.info(f"Robust scaling: {self.use_robust_scaling}")
            logger.info(f"CV folds: {self.cv_folds}")
            # X is now (N, 3)
            # Let's log the range for each feature
            logger.info(f"Input features shape: {X.shape}")
            # We assume X is now a numerical numpy array
            try:
                logger.info(f"  Feature 1 (stoi) range:         [{X[:, 0].min():.4f}, {X[:, 0].max():.4f}]")
                logger.info(f"  Feature 2 (n_words) range:      [{X[:, 1].min():.4f}, {X[:, 1].max():.4f}]")
                logger.info(f"  Feature 3 (hearing_loss) range: [{X[:, 2].min():.4f}, {X[:, 2].max():.4f}]")
            except Exception as e:
                logger.warning(f"Could not log feature ranges: {e}")

            logger.info(f"Correctness range: [{y.min():.4f}, {y.max():.4f}]")
                    
        # Step 1: Feature Engineering
        if self.verbose: logger.info("\n--- Step 1: Feature Engineering ---")
        X_engineered = self._engineer_features(X)
        
        # Step 2: Scaling
        if self.verbose: logger.info("\n--- Step 2: Feature Scaling ---")
        X_scaled = self.scaler.fit_transform(X_engineered)
        
        # Step 3: Hyperparameter Tuning
        if self.verbose: logger.info("\n--- Step 3: Hyperparameter Tuning ---")
        best_params = self._hyperparameter_tuning(X_scaled, y)
        
        # Step 4: Train Final Ensemble Models
        if self.verbose: logger.info("\n--- Step 4: Training Final Ensemble ---")
        
        # --- CRITICAL FIX: Increased max_iter ---
        self.ridge_model = Ridge(alpha=best_params['ridge_alpha'], max_iter=100000)
        self.ridge_model.fit(X_scaled, y)
        
        self.lasso_model = Lasso(alpha=best_params['lasso_alpha'], max_iter=100000)
        self.lasso_model.fit(X_scaled, y)
        
        self.elastic_model = ElasticNet(
            alpha=best_params['elastic_alpha'],
            l1_ratio=best_params['elastic_l1_ratio'],
            max_iter=100000
        )
        self.elastic_model.fit(X_scaled, y)
        
        # Calculate final training RMSE
        w1, w2, w3 = self.ensemble_weights
        ensemble_pred = (w1 * self.ridge_model.predict(X_scaled) +
                         w2 * self.lasso_model.predict(X_scaled) +
                         w3 * self.elastic_model.predict(X_scaled))
        ensemble_pred = np.clip(ensemble_pred, 0, 100)
        
        self.train_rmse = np.sqrt(np.mean((y - ensemble_pred) ** 2))
        self.fitted = True
        
        if self.verbose:
            logger.info(f"Ridge trained - Training RMSE: {np.sqrt(np.mean((y - self.ridge_model.predict(X_scaled)) ** 2)):.4f}")
            logger.info(f"Lasso trained - Training RMSE: {np.sqrt(np.mean((y - self.lasso_model.predict(X_scaled)) ** 2)):.4f}")
            logger.info(f"ElasticNet trained - Training RMSE: {np.sqrt(np.mean((y - self.elastic_model.predict(X_scaled)) ** 2)):.4f}")
            logger.info(f"\nENSEMBLE TRAINING RMSE: {self.train_rmse:.4f}")
            logger.info(f"Fitting {self.model_name} complete.")
            logger.info(f"=" * 60)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict correctness values from STOI scores using ensemble.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        
        # Feature engineering
        X_engineered = self._engineer_features(X)
        
        # Scaling
        X_scaled = self.scaler.transform(X_engineered)
        
        # Ensemble prediction
        w1, w2, w3 = self.ensemble_weights
        ridge_pred = self.ridge_model.predict(X_scaled)
        lasso_pred = self.lasso_model.predict(X_scaled)
        elastic_pred = self.elastic_model.predict(X_scaled)
        
        predictions = (w1 * ridge_pred + w2 * lasso_pred + w3 * elastic_pred)
        
        # Clip to valid range
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def get_params(self) -> dict:
        """Get model parameters and statistics."""
        if not self.fitted:
            return {'fitted': False, 'model_name': self.model_name}
        
        return {
            'model_name': self.model_name,
            'fitted': True,
            'poly_degree': self.poly_degree,
            'use_robust_scaling': self.use_robust_scaling,
            'ensemble_weights': self.ensemble_weights,
            'train_rmse': self.train_rmse,
            'n_features_engineered': self.n_features_engineered,
            'ridge_alpha': self.ridge_model.alpha,
            'lasso_alpha': self.lasso_model.alpha,
            'elastic_alpha': self.elastic_model.alpha,
            'elastic_l1_ratio': self.elastic_model.l1_ratio
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model to a pickle file."""
        if not self.fitted:
            logger.warning("Saving unfitted model")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        if self.verbose:
            logger.info(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'ProposedEnhancedModel':
        """Load a fitted model from a pickle file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from: {filepath}")
        return model
    
    def __repr__(self) -> str:
        if self.fitted:
            return (f"{self.model_name}(fitted=True, poly_degree={self.poly_degree}, "
                    f"train_rmse={self.train_rmse:.4f})")
        return f"{self.model_name}(fitted=False)"

#
# NOTE: The `if __name__ == "__main__":` block (the fake data test)
# has been removed. This file is now *only* a class blueprint,
# ready to be imported by `run_experiments.py`.
#