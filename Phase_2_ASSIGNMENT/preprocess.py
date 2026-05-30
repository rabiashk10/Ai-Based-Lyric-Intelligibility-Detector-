"""
Preprocessing Module for Cadenza CLIP1 Challenge
Handles data splitting, normalization, and preparation for training.

This module:
1. Splits training data into train/test subsets
2. Normalizes features (optional)
3. Handles data validation and cleaning
4. Prepares data for model training and evaluation
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing class for STOI baseline models.
    
    Handles:
    - Train/test splitting
    - Feature scaling (optional)
    - Data validation
    - Feature extraction
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, 
                 normalize: bool = False):
        """
        Initialize the preprocessor.
        
        Args:
            test_size (float): Proportion of data for testing (default: 0.2 = 20%)
            random_state (int): Random seed for reproducibility (default: 42)
            normalize (bool): Whether to normalize features (default: False)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.is_fitted = False
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train and test sets.
        
        Args:
            df (pd.DataFrame): Input dataframe with features and labels
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        logger.info(f"Splitting data: {len(df)} samples total")
        logger.info(f"Test size: {self.test_size * 100:.0f}%")
        logger.info(f"Random state: {self.random_state}")
        
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        logger.info(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, test_df
    
    def extract_features_and_labels(
        self, 
        df: pd.DataFrame,
        feature_col: str = 'stoi',
        target_col: str = 'correctness'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features (X) and labels (y) from dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_col (str): Name of feature column (default: 'stoi')
            target_col (str): Name of target column (default: 'correctness')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) features and labels
            
        Raises:
            KeyError: If required columns are missing
        """
        # Validate columns exist
        if feature_col not in df.columns:
            raise KeyError(f"Feature column '{feature_col}' not found in dataframe. "
                         f"Available columns: {list(df.columns)}")
        
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe. "
                         f"Available columns: {list(df.columns)}")
        
        # Extract arrays
        X = df[feature_col].values
        y = df[target_col].values
        
        # Validate data
        if np.any(np.isnan(X)):
            n_nan = np.sum(np.isnan(X))
            logger.warning(f"Found {n_nan} NaN values in features. Removing...")
            valid_mask = ~np.isnan(X) & ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
        
        logger.info(f"Extracted features: shape={X.shape}, range=[{X.min():.4f}, {X.max():.4f}]")
        logger.info(f"Extracted labels: shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}]")
        
        return X, y
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        Args:
            X (np.ndarray): Input features
            fit (bool): If True, fit the scaler. If False, use existing fit.
            
        Returns:
            np.ndarray: Normalized features
        """
        if not self.normalize:
            logger.info("Normalization disabled, returning original features")
            return X
        
        # Reshape if 1D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
            logger.info(f"Fitted scaler: mean={self.scaler.mean_[0]:.4f}, "
                       f"std={self.scaler.scale_[0]:.4f}")
        else:
            if not self.is_fitted:
                raise RuntimeError("Scaler not fitted yet. Call normalize_features with fit=True first.")
            X_scaled = self.scaler.transform(X)
            logger.info("Applied existing scaler transformation")
        
        # Flatten back to 1D if needed
        if X_scaled.shape[1] == 1:
            X_scaled = X_scaled.ravel()
        
        return X_scaled
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_col: str = 'stoi',
        target_col: str = 'correctness',
        split: bool = True
    ) -> dict:
        """
        Complete data preparation pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_col (str): Feature column name
            target_col (str): Target column name
            split (bool): Whether to split into train/test
            
        Returns:
            dict: Dictionary with prepared data
                If split=True: {'X_train', 'X_test', 'y_train', 'y_test', 
                               'train_df', 'test_df'}
                If split=False: {'X', 'y', 'df'}
        """
        logger.info("="*60)
        logger.info("DATA PREPROCESSING PIPELINE")
        logger.info("="*60)
        logger.info(f"Input data: {len(df)} samples")
        logger.info(f"Feature column: {feature_col}")
        logger.info(f"Target column: {target_col}")
        logger.info(f"Normalization: {self.normalize}")
        logger.info(f"Split data: {split}")
        
        if split:
            # Split into train and test
            train_df, test_df = self.split_data(df)
            
            # Extract features and labels
            X_train, y_train = self.extract_features_and_labels(
                train_df, feature_col, target_col
            )
            X_test, y_test = self.extract_features_and_labels(
                test_df, feature_col, target_col
            )
            
            # Normalize if requested
            if self.normalize:
                logger.info("Normalizing features...")
                X_train = self.normalize_features(X_train, fit=True)
                X_test = self.normalize_features(X_test, fit=False)
            
            logger.info("="*60)
            logger.info("PREPROCESSING COMPLETE")
            logger.info("="*60)
            logger.info(f"Training data: {len(X_train)} samples")
            logger.info(f"Test data: {len(X_test)} samples")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_df': train_df,
                'test_df': test_df
            }
        else:
            # No splitting, return full dataset
            X, y = self.extract_features_and_labels(df, feature_col, target_col)
            
            if self.normalize:
                logger.info("Normalizing features...")
                X = self.normalize_features(X, fit=True)
            
            logger.info("="*60)
            logger.info("PREPROCESSING COMPLETE")
            logger.info("="*60)
            logger.info(f"Total data: {len(X)} samples")
            
            return {
                'X': X,
                'y': y,
                'df': df
            }
    
    def get_statistics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute statistics for features and labels.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            dict: Statistics dictionary
        """
        stats = {
            'n_samples': len(X),
            'feature_mean': float(np.mean(X)),
            'feature_std': float(np.std(X)),
            'feature_min': float(np.min(X)),
            'feature_max': float(np.max(X)),
            'label_mean': float(np.mean(y)),
            'label_std': float(np.std(y)),
            'label_min': float(np.min(y)),
            'label_max': float(np.max(y))
        }
        return stats


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = False
) -> dict:
    """
    Convenience function to prepare training data.
    
    Args:
        df (pd.DataFrame): Input dataframe with training data
        test_size (float): Proportion for test set
        random_state (int): Random seed
        normalize (bool): Whether to normalize features
        
    Returns:
        dict: Prepared data dictionary
    """
    preprocessor = DataPreprocessor(
        test_size=test_size,
        random_state=random_state,
        normalize=normalize
    )
    
    return preprocessor.prepare_data(df, split=True)


def prepare_validation_data(
    df: pd.DataFrame,
    feature_col: str = 'stoi',
    normalize: bool = False,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Prepare validation data (features only, no labels).
    
    Args:
        df (pd.DataFrame): Validation dataframe
        feature_col (str): Feature column name
        normalize (bool): Whether to normalize
        scaler (StandardScaler, optional): Pre-fitted scaler from training
        
    Returns:
        np.ndarray: Prepared features
    """
    logger.info("Preparing validation data (features only)...")
    
    if feature_col not in df.columns:
        raise KeyError(f"Feature column '{feature_col}' not found")
    
    X = df[feature_col].values
    
    # Remove NaN values
    if np.any(np.isnan(X)):
        n_nan = np.sum(np.isnan(X))
        logger.warning(f"Found {n_nan} NaN values in validation features. Removing...")
        X = X[~np.isnan(X)]
    
    # Normalize if requested
    if normalize and scaler is not None:
        X = X.reshape(-1, 1)
        X = scaler.transform(X)
        X = X.ravel()
        logger.info("Applied normalization using provided scaler")
    
    logger.info(f"Validation features: {len(X)} samples, range=[{X.min():.4f}, {X.max():.4f}]")
    
    return X


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_df = pd.DataFrame({
        'signal': [f'signal_{i}' for i in range(n_samples)],
        'stoi': np.random.beta(2, 2, n_samples),
        'correctness': np.random.uniform(0, 1, n_samples),
        'hearing_loss': np.random.choice(['Mild', 'Moderate', 'Severe'], n_samples)
    })
    
    print("\n" + "="*60)
    print("TESTING PREPROCESSING MODULE")
    print("="*60)
    
    # Test 1: Basic split without normalization
    print("\n--- Test 1: Basic Split (No Normalization) ---")
    preprocessor1 = DataPreprocessor(test_size=0.2, random_state=42, normalize=False)
    data1 = preprocessor1.prepare_data(synthetic_df)
    
    print(f"\nX_train shape: {data1['X_train'].shape}")
    print(f"X_test shape: {data1['X_test'].shape}")
    print(f"y_train shape: {data1['y_train'].shape}")
    print(f"y_test shape: {data1['y_test'].shape}")
    
    # Test 2: Split with normalization
    print("\n--- Test 2: Split with Normalization ---")
    preprocessor2 = DataPreprocessor(test_size=0.2, random_state=42, normalize=True)
    data2 = preprocessor2.prepare_data(synthetic_df)
    
    print(f"\nX_train (normalized) - mean: {data2['X_train'].mean():.4f}, std: {data2['X_train'].std():.4f}")
    print(f"X_test (normalized) - mean: {data2['X_test'].mean():.4f}, std: {data2['X_test'].std():.4f}")
    
    # Test 3: Statistics
    print("\n--- Test 3: Statistics ---")
    stats = preprocessor1.get_statistics(data1['X_train'], data1['y_train'])
    print("\nTraining data statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 4: Convenience function
    print("\n--- Test 4: Convenience Function ---")
    data3 = prepare_training_data(synthetic_df, test_size=0.3, random_state=123)
    print(f"Using convenience function - Train: {len(data3['X_train'])}, Test: {len(data3['X_test'])}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)