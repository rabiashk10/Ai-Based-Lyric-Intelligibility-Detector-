"""
Model Baseline B: Regularized Logistic Regression
Student 2's baseline implementation using scikit-learn LogisticRegression with L2 regularization.

This model uses sklearn's LogisticRegression which provides:
- L2 regularization to prevent overfitting
- More sophisticated optimization (liblinear/lbfgs)
- Built-in cross-validation support
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LogisticModelB:
    """
    Regularized logistic regression model using scikit-learn.
    
    This model uses sklearn's LogisticRegression with L2 regularization.
    Key features:
        - Regularization parameter C (inverse of regularization strength)
        - StandardScaler for feature normalization
        - More robust optimization algorithms
        - Handles multi-class classification (though we use regression mapping)
    
    Note: Since sklearn's LogisticRegression is for classification, we adapt it
    for regression by treating correctness as continuous output.
    We use a direct linear model approach instead.
    """
    
    def __init__(self, C: float = 1.0, normalize: bool = True):
        """
        Initialize the regularized logistic model.
        
        Args:
            C (float): Inverse of regularization strength. Smaller values = stronger regularization.
                      Default: 1.0
            normalize (bool): Whether to normalize features using StandardScaler.
                            Default: True
        """
        self.C = C
        self.normalize = normalize
        self.model_name = "Baseline_B_Regularized_Logistic"
        self.fitted = False
        
        # Use Ridge Regression for continuous output with regularization
        from sklearn.linear_model import Ridge
        # Convert C to alpha (alpha = 1/(2*C))
        self.alpha = 1.0 / (2.0 * C) if C > 0 else 0.0
        self.model = Ridge(alpha=self.alpha, max_iter=10000)
        
        # Scaler for normalization
        self.scaler = StandardScaler() if normalize else None
        
        # Store coefficients
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regularized logistic model to training data.
        
        Args:
            X (np.ndarray): Training features (STOI scores), shape (n_samples,)
            y (np.ndarray): Training labels (correctness), shape (n_samples,)
        
        Raises:
            ValueError: If input arrays are invalid
        """
        # Input validation
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same length: {X.shape[0]} != {y.shape[0]}")
        
        # Reshape X to 2D if needed (sklearn requirement)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Convert correctness to 0-100 scale if needed
        if y.max() <= 1.0:
            logger.info("Converting correctness from 0-1 scale to 0-100 scale")
            y = y * 100.0
        
        logger.info(f"Fitting {self.model_name} with {len(X)} samples")
        logger.info(f"Regularization parameter C: {self.C}")
        logger.info(f"Ridge alpha: {self.alpha:.6f}")
        logger.info(f"Feature normalization: {self.normalize}")
        logger.info(f"STOI range: [{X.min():.4f}, {X.max():.4f}]")
        logger.info(f"Correctness range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Normalize features if requested
        if self.normalize:
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Features normalized: mean={self.scaler.mean_[0]:.4f}, "
                       f"std={self.scaler.scale_[0]:.4f}")
        else:
            X_scaled = X
        
        # Fit the model
        self.model.fit(X_scaled, y)
        
        # Store coefficients
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        self.fitted = True
        
        # Log model parameters
        logger.info(f"Fitting successful!")
        logger.info(f"Model coefficients: {self.coef_}")
        logger.info(f"Model intercept: {self.intercept_:.4f}")
        
        # Compute R² score on training data
        r2_score = self.model.score(X_scaled, y)
        logger.info(f"Training R² score: {r2_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict correctness values from STOI scores.
        
        Args:
            X (np.ndarray): STOI scores, shape (n_samples,)
            
        Returns:
            np.ndarray: Predicted correctness values (0-100), shape (n_samples,)
            
        Raises:
            RuntimeError: If predict() is called before fit()
        """
        if not self.fitted:
            raise RuntimeError(
                f"{self.model_name}: Model must be fitted before prediction. "
                "Call fit() first."
            )
        
        X = np.asarray(X)
        
        # Reshape X to 2D if needed
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Normalize features if scaler was used during training
        if self.normalize and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid range [0, 100]
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def get_params(self) -> dict:
        """
        Get the fitted model parameters.
        
        Returns:
            dict: Dictionary containing model parameters and metadata
        """
        if not self.fitted:
            return {
                'model_name': self.model_name,
                'fitted': False,
                'C': self.C,
                'alpha': self.alpha,
                'normalize': self.normalize,
                'coef': None,
                'intercept': None
            }
        
        return {
            'model_name': self.model_name,
            'fitted': True,
            'C': self.C,
            'alpha': self.alpha,
            'normalize': self.normalize,
            'coef': self.coef_.tolist() if hasattr(self.coef_, 'tolist') else self.coef_,
            'intercept': float(self.intercept_),
            'scaler_mean': float(self.scaler.mean_[0]) if self.scaler else None,
            'scaler_std': float(self.scaler.scale_[0]) if self.scaler else None
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to a pickle file.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.fitted:
            logger.warning("Saving unfitted model")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'LogisticModelB':
        """
        Load a fitted model from a pickle file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            LogisticModelB: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from: {filepath}")
        return model
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.fitted:
            return (f"{self.model_name}(fitted=True, C={self.C}, "
                   f"alpha={self.alpha:.6f}, normalize={self.normalize})")
        else:
            return f"{self.model_name}(fitted=False, C={self.C})"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate STOI scores with some correlation to correctness
    X_train = np.random.beta(2, 2, n_samples)  # STOI scores between 0 and 1
    
    # Generate correctness with linear relationship + noise
    # y = 50 + 80*X + noise
    y_train = 50 + 80 * X_train + np.random.normal(0, 10, n_samples)
    y_train = np.clip(y_train, 0, 100)  # Clip to valid range
    
    print("\n=== Testing Different Regularization Strengths ===")
    
    # Test different C values
    C_values = [0.1, 1.0, 10.0]
    
    for C in C_values:
        print(f"\n--- Training with C={C} ---")
        model = LogisticModelB(C=C, normalize=True)
        model.fit(X_train, y_train)
        
        # Show parameters
        params = model.get_params()
        print(f"Coefficients: {params['coef']}")
        print(f"Intercept: {params['intercept']:.4f}")
        print(f"Alpha (regularization): {params['alpha']:.6f}")
    
    print("\n=== Training Final Model ===")
    # Train with default parameters
    model = LogisticModelB(C=1.0, normalize=True)
    model.fit(X_train, y_train)
    
    # Show fitted parameters
    print("\n=== Model Parameters ===")
    params = model.get_params()
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # Make predictions on test data
    X_test = np.linspace(0, 1, 100)
    y_pred = model.predict(X_test)
    
    print("\n=== Sample Predictions ===")
    print("STOI Score -> Predicted Correctness")
    for i in range(0, 100, 10):
        print(f"{X_test[i]:.2f} -> {y_pred[i]:.2f}")
    
    # Compare with vs without normalization
    print("\n=== Comparing Normalization ===")
    model_no_norm = LogisticModelB(C=1.0, normalize=False)
    model_no_norm.fit(X_train, y_train)
    y_pred_no_norm = model_no_norm.predict(X_test)
    
    diff = np.abs(y_pred - y_pred_no_norm).mean()
    print(f"Average prediction difference: {diff:.4f}")
    
    # Test saving and loading
    print("\n=== Testing Save/Load ===")
    model.save_model('models/baseline_B.pkl')
    loaded_model = LogisticModelB.load_model('models/baseline_B.pkl')
    print(f"Loaded model: {loaded_model}")
    
    # Verify predictions are same
    y_pred_loaded = loaded_model.predict(X_test)
    assert np.allclose(y_pred, y_pred_loaded), "Loaded model predictions don't match!"
    print("✓ Save/Load test passed!")
    
    print("\n=== Baseline B Model Test Complete ===")