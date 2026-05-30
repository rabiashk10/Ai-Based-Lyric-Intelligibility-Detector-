"""
Model Baseline A: Standard Logistic Regression
Student 1's baseline implementation using logistic curve fitting.

This model maps STOI scores (0-1) to correctness predictions (0-100) 
using a logistic (sigmoid) function with two parameters.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class LogisticModelA:
    """
    Standard logistic regression model for intelligibility prediction.
    
    Uses a logistic (sigmoid) curve to map STOI scores to correctness values.
    The model fits two parameters:
        - x_0: The midpoint of the sigmoid (inflection point)
        - k: The steepness/growth rate of the curve
    
    Mathematical form: y = 100 / (1 + exp(-k * (x - x_0)))
    """
    
    def __init__(self):
        """Initialize the logistic model."""
        self.params: Optional[np.ndarray] = None
        self.model_name = "Baseline_A_Standard_Logistic"
        self.fitted = False
        
    def _logistic_mapping(self, x: np.ndarray, x_0: float, k: float) -> np.ndarray:
        """
        Logistic (sigmoid) function.
        
        Args:
            x (np.ndarray): Input values (STOI scores)
            x_0 (float): Midpoint parameter (where output = 50)
            k (float): Growth rate parameter (steepness of curve)
            
        Returns:
            np.ndarray: Predicted correctness values (0-100)
        """
        return 100.0 / (1.0 + np.exp(-k * (x - x_0)))
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            initial_guess: Optional[Tuple[float, float]] = None) -> None:
        """
        Fit the logistic model to training data.
        
        Args:
            X (np.ndarray): Training features (STOI scores), shape (n_samples,)
            y (np.ndarray): Training labels (correctness), shape (n_samples,)
            initial_guess (tuple, optional): Initial parameter values (x_0, k).
                                            Defaults to (0.5, 1.0)
        
        Raises:
            ValueError: If input arrays are invalid
        """
        # Input validation
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same length: {X.shape[0]} != {y.shape[0]}")
        
        if len(X.shape) != 1:
            raise ValueError(f"X must be 1-dimensional, got shape {X.shape}")
        
        # Convert correctness to 0-100 scale if needed
        if y.max() <= 1.0:
            logger.info("Converting correctness from 0-1 scale to 0-100 scale")
            y = y * 100.0
        
        # Set initial parameter guess
        if initial_guess is None:
            initial_guess = (0.5, 1.0)  # Default: midpoint at 0.5, moderate steepness
        
        logger.info(f"Fitting {self.model_name} with {len(X)} samples")
        logger.info(f"STOI range: [{X.min():.4f}, {X.max():.4f}]")
        logger.info(f"Correctness range: [{y.min():.4f}, {y.max():.4f}]")
        logger.info(f"Initial guess: x_0={initial_guess[0]}, k={initial_guess[1]}")
        
        try:
            # Fit logistic curve using scipy's curve_fit
            self.params, pcov = curve_fit(
                self._logistic_mapping, 
                X, 
                y, 
                p0=initial_guess,
                maxfev=10000  # Maximum function evaluations
            )
            
            self.fitted = True
            
            # Log fitted parameters
            logger.info(f"Fitting successful!")
            logger.info(f"Fitted parameters: x_0={self.params[0]:.4f}, k={self.params[1]:.4f}")
            logger.info(f"Parameter covariance matrix diagonal: {np.diag(pcov)}")
            
        except RuntimeError as e:
            logger.error(f"Curve fitting failed: {e}")
            raise
    
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
        if not self.fitted or self.params is None:
            raise RuntimeError(
                f"{self.model_name}: Model must be fitted before prediction. "
                "Call fit() first."
            )
        
        X = np.asarray(X)
        
        if len(X.shape) != 1:
            raise ValueError(f"X must be 1-dimensional, got shape {X.shape}")
        
        # Make predictions
        predictions = self._logistic_mapping(X, self.params[0], self.params[1])
        
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
                'params': None
            }
        
        return {
            'model_name': self.model_name,
            'fitted': True,
            'x_0': float(self.params[0]),
            'k': float(self.params[1]),
            'params': self.params.tolist()
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
    def load_model(filepath: str) -> 'LogisticModelA':
        """
        Load a fitted model from a pickle file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            LogisticModelA: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from: {filepath}")
        return model
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.fitted:
            return (f"{self.model_name}(fitted=True, "
                   f"x_0={self.params[0]:.4f}, k={self.params[1]:.4f})")
        else:
            return f"{self.model_name}(fitted=False)"


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
    
    # True parameters for synthetic data
    true_x0 = 0.6
    true_k = 8.0
    
    # Generate STOI scores
    X_train = np.random.beta(2, 2, n_samples)  # STOI scores between 0 and 1
    
    # Generate true correctness with some noise
    y_train = 100.0 / (1.0 + np.exp(-true_k * (X_train - true_x0)))
    y_train += np.random.normal(0, 5, n_samples)  # Add noise
    y_train = np.clip(y_train, 0, 100)  # Clip to valid range
    
    # Create and train model
    print("\n=== Training Baseline A Model ===")
    model = LogisticModelA()
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
    
    # Test saving and loading
    print("\n=== Testing Save/Load ===")
    model.save_model('models/baseline_A.pkl')
    loaded_model = LogisticModelA.load_model('models/baseline_A.pkl')
    print(f"Loaded model: {loaded_model}")
    
    # Verify predictions are same
    y_pred_loaded = loaded_model.predict(X_test)
    assert np.allclose(y_pred, y_pred_loaded), "Loaded model predictions don't match!"
    print("✓ Save/Load test passed!")
    
    print("\n=== Baseline A Model Test Complete ===")