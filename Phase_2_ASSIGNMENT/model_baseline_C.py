"""
Model Baseline C: Polynomial Regression
Student 3's baseline implementation using polynomial features with Ridge regression.

This model uses polynomial feature transformation to capture non-linear relationships:
- Transforms STOI → [1, STOI, STOI², STOI³, ...]
- Applies Ridge regression for stability
- Can capture curved relationships better than linear models
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class PolynomialModelC:
    """
    Polynomial regression model for intelligibility prediction.
    
    This model creates polynomial features from STOI scores and fits
    a regularized regression model. For example, with degree=2:
    - Input: [STOI]
    - Features: [1, STOI, STOI²]
    - Output: w0 + w1*STOI + w2*STOI²
    
    Key features:
        - Polynomial feature transformation (degree 2 or 3)
        - Ridge regression with L2 regularization
        - StandardScaler for feature normalization
        - Can capture non-linear relationships
    """
    
    def __init__(self, degree: int = 2, alpha: float = 1.0, normalize: bool = True):
        """
        Initialize the polynomial regression model.
        
        Args:
            degree (int): Degree of polynomial features. 
                         degree=2 creates [1, x, x²]
                         degree=3 creates [1, x, x², x³]
                         Default: 2
            alpha (float): Regularization strength for Ridge regression.
                          Larger values = more regularization.
                          Default: 1.0
            normalize (bool): Whether to normalize features using StandardScaler.
                            Default: True
        """
        self.degree = degree
        self.alpha = alpha
        self.normalize = normalize
        self.model_name = f"Baseline_C_Polynomial_Degree{degree}"
        self.fitted = False
        
        # Create polynomial features transformer
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            include_bias=True  # Includes constant term
        )
        
        # Create Ridge regression model
        self.ridge = Ridge(alpha=alpha, max_iter=10000)
        
        # Create scaler for normalization (optional)
        self.scaler = StandardScaler() if normalize else None
        
        # Store coefficients
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the polynomial regression model to training data.
        
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
        
        # Reshape X to 2D if needed
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Convert correctness to 0-100 scale if needed
        if y.max() <= 1.0:
            logger.info("Converting correctness from 0-1 scale to 0-100 scale")
            y = y * 100.0
        
        logger.info(f"Fitting {self.model_name} with {len(X)} samples")
        logger.info(f"Polynomial degree: {self.degree}")
        logger.info(f"Regularization alpha: {self.alpha}")
        logger.info(f"Feature normalization: {self.normalize}")
        logger.info(f"STOI range: [{X.min():.4f}, {X.max():.4f}]")
        logger.info(f"Correctness range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Step 1: Create polynomial features
        X_poly = self.poly_features.fit_transform(X)
        self.feature_names_ = self.poly_features.get_feature_names_out(['STOI'])
        
        logger.info(f"Created {X_poly.shape[1]} polynomial features: {self.feature_names_}")
        logger.info(f"Feature matrix shape: {X_poly.shape}")
        
        # Step 2: Normalize features if requested
        if self.normalize:
            X_scaled = self.scaler.fit_transform(X_poly)
            logger.info(f"Features normalized")
        else:
            X_scaled = X_poly
        
        # Step 3: Fit Ridge regression
        self.ridge.fit(X_scaled, y)
        
        # Store coefficients
        self.coef_ = self.ridge.coef_
        self.intercept_ = self.ridge.intercept_
        
        self.fitted = True
        
        # Log model parameters
        logger.info(f"Fitting successful!")
        logger.info(f"Model intercept: {self.intercept_:.4f}")
        logger.info(f"Model coefficients: {self.coef_}")
        
        # Show polynomial equation
        equation = self._get_polynomial_equation()
        logger.info(f"Polynomial equation: {equation}")
        
        # Compute R² score on training data
        r2_score = self.ridge.score(X_scaled, y)
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
        
        # Transform to polynomial features
        X_poly = self.poly_features.transform(X)
        
        # Normalize features if scaler was used during training
        if self.normalize and self.scaler is not None:
            X_scaled = self.scaler.transform(X_poly)
        else:
            X_scaled = X_poly
        
        # Make predictions
        predictions = self.ridge.predict(X_scaled)
        
        # Clip to valid range [0, 100]
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def _get_polynomial_equation(self) -> str:
        """
        Generate a string representation of the polynomial equation.
        
        Returns:
            str: Polynomial equation as a string
        """
        if not self.fitted:
            return "Model not fitted"
        
        terms = [f"{self.intercept_:.4f}"]
        
        for i, (coef, name) in enumerate(zip(self.coef_, self.feature_names_)):
            if abs(coef) > 1e-6:  # Only show non-zero coefficients
                sign = "+" if coef >= 0 else "-"
                terms.append(f"{sign} {abs(coef):.4f}*{name}")
        
        return " ".join(terms)
    
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
                'degree': self.degree,
                'alpha': self.alpha,
                'normalize': self.normalize,
                'coef': None,
                'intercept': None
            }
        
        return {
            'model_name': self.model_name,
            'fitted': True,
            'degree': self.degree,
            'alpha': self.alpha,
            'normalize': self.normalize,
            'n_features': len(self.feature_names_),
            'feature_names': self.feature_names_.tolist(),
            'coef': self.coef_.tolist(),
            'intercept': float(self.intercept_),
            'equation': self._get_polynomial_equation()
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
    def load_model(filepath: str) -> 'PolynomialModelC':
        """
        Load a fitted model from a pickle file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            PolynomialModelC: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from: {filepath}")
        return model
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.fitted:
            return (f"{self.model_name}(fitted=True, degree={self.degree}, "
                   f"alpha={self.alpha}, n_features={len(self.feature_names_)})")
        else:
            return f"{self.model_name}(fitted=False, degree={self.degree})"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate synthetic test data with non-linear relationship
    np.random.seed(42)
    n_samples = 1000
    
    # Generate STOI scores
    X_train = np.random.beta(2, 2, n_samples)  # STOI scores between 0 and 1
    
    # Generate correctness with quadratic relationship + noise
    # y = 20 + 30*X + 50*X² + noise
    y_train = 20 + 30 * X_train + 50 * (X_train ** 2) + np.random.normal(0, 8, n_samples)
    y_train = np.clip(y_train, 0, 100)  # Clip to valid range
    
    print("\n=== Testing Different Polynomial Degrees ===")
    
    # Test different degrees
    degrees = [1, 2, 3]
    
    for degree in degrees:
        print(f"\n--- Training with degree={degree} ---")
        model = PolynomialModelC(degree=degree, alpha=1.0, normalize=True)
        model.fit(X_train, y_train)
        
        # Show equation
        params = model.get_params()
        print(f"Polynomial equation:\n{params['equation']}")
    
    print("\n=== Training Final Model ===")
    # Train with default parameters (degree=2)
    model = PolynomialModelC(degree=2, alpha=1.0, normalize=True)
    model.fit(X_train, y_train)
    
    # Show fitted parameters
    print("\n=== Model Parameters ===")
    params = model.get_params()
    for key, value in params.items():
        if key != 'coef':  # Skip long coefficient list
            print(f"{key}: {value}")
    
    # Make predictions on test data
    X_test = np.linspace(0, 1, 100)
    y_pred = model.predict(X_test)
    
    print("\n=== Sample Predictions ===")
    print("STOI Score -> Predicted Correctness")
    for i in range(0, 100, 10):
        print(f"{X_test[i]:.2f} -> {y_pred[i]:.2f}")
    
    # Compare different regularization strengths
    print("\n=== Comparing Regularization Strengths ===")
    alphas = [0.1, 1.0, 10.0]
    
    for alpha in alphas:
        model_alpha = PolynomialModelC(degree=2, alpha=alpha, normalize=True)
        model_alpha.fit(X_train, y_train)
        y_pred_alpha = model_alpha.predict(X_test)
        
        # Compute RMSE on test data (using true quadratic relationship)
        y_true_test = 20 + 30 * X_test + 50 * (X_test ** 2)
        y_true_test = np.clip(y_true_test, 0, 100)
        rmse = np.sqrt(np.mean((y_pred_alpha - y_true_test) ** 2))
        print(f"Alpha={alpha:4.1f}: RMSE={rmse:.4f}")
    
    # Test saving and loading
    print("\n=== Testing Save/Load ===")
    model.save_model('models/baseline_C.pkl')
    loaded_model = PolynomialModelC.load_model('models/baseline_C.pkl')
    print(f"Loaded model: {loaded_model}")
    
    # Verify predictions are same
    y_pred_loaded = loaded_model.predict(X_test)
    assert np.allclose(y_pred, y_pred_loaded), "Loaded model predictions don't match!"
    print("✓ Save/Load test passed!")
    
    print("\n=== Baseline C Model Test Complete ===")