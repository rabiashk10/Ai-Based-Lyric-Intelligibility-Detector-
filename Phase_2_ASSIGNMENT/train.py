"""
Training Pipeline for Cadenza CLIP1 Challenge
Trains all three baseline models and generates predictions.


This script:
1. Loads ONLY training data (with correctness labels)
2. Splits training data into train/test (80/20)
3. Trains Baseline A, B, and C models
4. Evaluates on test split (where we have labels!)
5. Optionally makes predictions on validation set (for leaderboard)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_loader import load_dataset_with_score
from preprocess import prepare_training_data, prepare_validation_data
from model_baseline_A import LogisticModelA
from model_baseline_B import LogisticModelB
from model_baseline_C import PolynomialModelC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Data paths
    DATA_ROOT = "C:/Code/AISEMPROJECT"
    DATASET = "cadenza_data"
    SYSTEM = "stoi"
    SCORES_DIR = "precomputed"
    
    # Preprocessing
    TEST_SIZE = 0.2          # 20% of training data for testing
    RANDOM_STATE = 42        # For reproducibility
    
    # Output directories
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    
    # Model hyperparameters
    BASELINE_A_PARAMS = {}  # Uses defaults
    
    BASELINE_B_PARAMS = {
        'C': 1.0,
        'normalize': True
    }
    
    BASELINE_C_PARAMS = {
        'degree': 2,
        'alpha': 1.0,
        'normalize': True
    }
    
    # Whether to generate validation predictions for leaderboard
    GENERATE_VALIDATION_PREDICTIONS = True


def setup_directories():
    """Create necessary output directories."""
    directories = [
        TrainingConfig.MODELS_DIR,
        TrainingConfig.RESULTS_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory ready: {directory}/")


def load_and_prepare_data() -> dict:
    """
    Load training data and split into train/test.
    
    Returns:
        dict: Prepared data with X_train, X_test, y_train, y_test
    """
    logger.info("="*60)
    logger.info("LOADING AND PREPROCESSING DATA")
    logger.info("="*60)
    
    # Load ONLY training data (has correctness labels!)
    logger.info("Loading training data...")
    train_df = load_dataset_with_score(
        data_root=TrainingConfig.DATA_ROOT,
        dataset=TrainingConfig.DATASET,
        split='train',  # Only train - it has correctness!
        system=TrainingConfig.SYSTEM,
        scores_dir=TrainingConfig.SCORES_DIR
    )
    
    logger.info(f"Loaded {len(train_df)} training samples")
    logger.info(f"Columns: {list(train_df.columns)}")
    
    # Verify correctness column exists
    if 'correctness' not in train_df.columns:
        raise ValueError("Training data must have 'correctness' column!")
    
    # Split training data into train/test using preprocessing
    logger.info("\nSplitting data using preprocessing module...")
    data = prepare_training_data(
        df=train_df,
        test_size=TrainingConfig.TEST_SIZE,
        random_state=TrainingConfig.RANDOM_STATE,
        normalize=False  # Models handle their own normalization
    )
    
    logger.info("\n✓ Data preparation complete!")
    logger.info(f"  Training set: {len(data['X_train'])} samples")
    logger.info(f"  Test set: {len(data['X_test'])} samples")
    
    return data


def train_baseline(model, model_name: str, X_train: np.ndarray, 
                   y_train: np.ndarray, X_test: np.ndarray) -> Tuple[object, np.ndarray, float]:
    """
    Train a baseline model and make predictions on test set.
    
    Args:
        model: Model instance to train
        model_name (str): Name of the model for logging
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        
    Returns:
        Tuple: (trained_model, predictions, training_time)
    """
    logger.info("\n" + "="*60)
    logger.info(f"TRAINING {model_name.upper()}")
    logger.info("="*60)
    
    # Train model
    logger.info(f"Training on {len(X_train)} samples...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"✓ Training completed in {training_time:.2f} seconds")
    
    # Make predictions on test set
    logger.info(f"Making predictions on {len(X_test)} test samples...")
    predictions = model.predict(X_test)
    
    logger.info(f"✓ Predictions complete")
    logger.info(f"  Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return model, predictions, training_time


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions (np.ndarray): Predicted values
        labels (np.ndarray): Ground truth values
        
    Returns:
        Dict: Dictionary of metrics
    """
    from scipy.stats import pearsonr, kendalltau
    
    # Convert to 0-100 scale if needed
    if labels.max() <= 1.0:
        labels = labels * 100.0
    if predictions.max() <= 1.0:
        predictions = predictions * 100.0
    
    # Compute metrics
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    mae = np.mean(np.abs(predictions - labels))
    std_err = np.std(predictions - labels) / np.sqrt(len(predictions))
    ncc, _ = pearsonr(predictions, labels)
    kt, _ = kendalltau(predictions, labels)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Std': std_err,
        'NCC': ncc,
        'KT': kt
    }


def save_model_and_results(model, predictions: np.ndarray, test_df: pd.DataFrame,
                           y_test: np.ndarray, model_name: str, 
                           training_time: float, metrics: Dict) -> None:
    """
    Save trained model, test predictions, and metadata.
    
    Args:
        model: Trained model instance
        predictions (np.ndarray): Test predictions
        test_df (pd.DataFrame): Test DataFrame (for signal IDs)
        y_test (np.ndarray): Ground truth test labels
        model_name (str): Name of the model
        training_time (float): Training time in seconds
        metrics (Dict): Evaluation metrics
    """
    # Save model
    model_path = Path(TrainingConfig.MODELS_DIR) / f"{model_name}.pkl"
    model.save_model(str(model_path))
    logger.info(f"  ✓ Model saved: {model_path}")
    
    # Save test predictions with ground truth
    test_results_df = pd.DataFrame({
        'signal_ID': test_df['signal'].values,
        'ground_truth': y_test * 100 if y_test.max() <= 1.0 else y_test,
        'predicted': predictions
    })
    test_predictions_path = Path(TrainingConfig.RESULTS_DIR) / f"{model_name}_test_predictions.csv"
    test_results_df.to_csv(test_predictions_path, index=False)
    logger.info(f"  ✓ Test predictions saved: {test_predictions_path}")
    
    # Save training metadata with metrics
    metadata = {
        'model_name': model_name,
        'training_time_seconds': training_time,
        'model_parameters': model.get_params(),
        'test_metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'test_predictions_file': str(test_predictions_path),
        'model_file': str(model_path)
    }
    
    metadata_path = Path(TrainingConfig.RESULTS_DIR) / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Metadata saved: {metadata_path}")


def generate_validation_predictions(model, model_name: str):
    """
    Generate predictions on validation set for leaderboard submission.
    
    Args:
        model: Trained model
        model_name (str): Model name
    """
    logger.info(f"\n  Generating validation predictions for leaderboard...")
    
    # Load validation data (no correctness labels!)
    valid_df = load_dataset_with_score(
        data_root=TrainingConfig.DATA_ROOT,
        dataset=TrainingConfig.DATASET,
        split='valid',
        system=TrainingConfig.SYSTEM,
        scores_dir=TrainingConfig.SCORES_DIR
    )
    
    # Prepare validation features
    X_valid = prepare_validation_data(
        df=valid_df,
        feature_col='stoi',
        normalize=False
    )
    
    # Make predictions
    valid_predictions = model.predict(X_valid)
    
    # Save for leaderboard submission
    valid_submission_df = pd.DataFrame({
        'signal_ID': valid_df['signal'].values,
        'intelligibility_score': valid_predictions
    })
    
    submission_path = Path(TrainingConfig.RESULTS_DIR) / f"{model_name}_validation_submission.csv"
    valid_submission_df.to_csv(submission_path, index=False)
    logger.info(f"  ✓ Validation predictions saved: {submission_path}")


def main():
    """Main training pipeline."""
    
    print("\n" + "="*60)
    print("CADENZA CLIP1 TRAINING PIPELINE")
    print("="*60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    setup_directories()
    
    # Load and prepare data (split training data)
    data = load_and_prepare_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    test_df = data['test_df']
    
    # Store all results
    results_summary = {}
    
    # =====================================================================
    # BASELINE A: Standard Logistic Regression
    # =====================================================================
    
    print("\n" + "="*60)
    print("TRAINING BASELINE A - STANDARD LOGISTIC")
    print("="*60)
    
    model_A = LogisticModelA(**TrainingConfig.BASELINE_A_PARAMS)
    trained_A, pred_A, time_A = train_baseline(
        model_A, "baseline_A", X_train, y_train, X_test
    )
    
    # Evaluate on test set
    metrics_A = compute_metrics(pred_A, y_test)
    logger.info(f"\n✓ Baseline A Test Metrics:")
    for key, value in metrics_A.items():
        logger.info(f"    {key}: {value:.4f}")
    
    # Save everything
    save_model_and_results(trained_A, pred_A, test_df, y_test, 
                          "baseline_A", time_A, metrics_A)
    
    # Generate validation predictions if enabled
    if TrainingConfig.GENERATE_VALIDATION_PREDICTIONS:
        generate_validation_predictions(trained_A, "baseline_A")
    
    results_summary['baseline_A'] = {
        'training_time': time_A,
        'metrics': metrics_A
    }
    
    # =====================================================================
    # BASELINE B: Regularized Logistic Regression
    # =====================================================================
    
    print("\n" + "="*60)
    print("TRAINING BASELINE B - REGULARIZED LOGISTIC")
    print("="*60)
    
    model_B = LogisticModelB(**TrainingConfig.BASELINE_B_PARAMS)
    trained_B, pred_B, time_B = train_baseline(
        model_B, "baseline_B", X_train, y_train, X_test
    )
    
    # Evaluate on test set
    metrics_B = compute_metrics(pred_B, y_test)
    logger.info(f"\n✓ Baseline B Test Metrics:")
    for key, value in metrics_B.items():
        logger.info(f"    {key}: {value:.4f}")
    
    # Save everything
    save_model_and_results(trained_B, pred_B, test_df, y_test,
                          "baseline_B", time_B, metrics_B)
    
    # Generate validation predictions if enabled
    if TrainingConfig.GENERATE_VALIDATION_PREDICTIONS:
        generate_validation_predictions(trained_B, "baseline_B")
    
    results_summary['baseline_B'] = {
        'training_time': time_B,
        'metrics': metrics_B
    }
    
    # =====================================================================
    # BASELINE C: Polynomial Regression
    # =====================================================================
    
    print("\n" + "="*60)
    print("TRAINING BASELINE C - POLYNOMIAL REGRESSION")
    print("="*60)
    
    model_C = PolynomialModelC(**TrainingConfig.BASELINE_C_PARAMS)
    trained_C, pred_C, time_C = train_baseline(
        model_C, "baseline_C", X_train, y_train, X_test
    )
    
    # Evaluate on test set
    metrics_C = compute_metrics(pred_C, y_test)
    logger.info(f"\n✓ Baseline C Test Metrics:")
    for key, value in metrics_C.items():
        logger.info(f"    {key}: {value:.4f}")
    
    # Save everything
    save_model_and_results(trained_C, pred_C, test_df, y_test,
                          "baseline_C", time_C, metrics_C)
    
    # Generate validation predictions if enabled
    if TrainingConfig.GENERATE_VALIDATION_PREDICTIONS:
        generate_validation_predictions(trained_C, "baseline_C")
    
    results_summary['baseline_C'] = {
        'training_time': time_C,
        'metrics': metrics_C
    }
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY - MODEL COMPARISON")
    print("="*60)
    
    # Comparison table
    print(f"\n{'Model':<20} {'RMSE ↓':<10} {'NCC ↑':<10} {'KT ↑':<10} {'Time (s)':<10}")
    print("-" * 70)
    
    for model_name, results in results_summary.items():
        metrics = results['metrics']
        time_taken = results['training_time']
        print(
            f"{model_name:<20} "
            f"{metrics['RMSE']:<10.4f} "
            f"{metrics['NCC']:<10.4f} "
            f"{metrics['KT']:<10.4f} "
            f"{time_taken:<10.2f}"
        )
    
    # Find best model
    best_rmse_model = min(results_summary.items(), key=lambda x: x[1]['metrics']['RMSE'])
    best_ncc_model = max(results_summary.items(), key=lambda x: x[1]['metrics']['NCC'])
    
    print("\n" + "-" * 70)
    print(f"Best RMSE: {best_rmse_model[0]} ({best_rmse_model[1]['metrics']['RMSE']:.4f})")
    print(f"Best NCC:  {best_ncc_model[0]} ({best_ncc_model[1]['metrics']['NCC']:.4f})")
    
    # Save summary
    summary_path = Path(TrainingConfig.RESULTS_DIR) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"\n✓ Training summary saved: {summary_path}")
    
    # Final message
    total_time = time_A + time_B + time_C
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Models saved in: {TrainingConfig.MODELS_DIR}/")
    print(f"Results saved in: {TrainingConfig.RESULTS_DIR}/")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ All 3 models trained successfully!")
    print("✓ Test predictions and metrics saved")
    if TrainingConfig.GENERATE_VALIDATION_PREDICTIONS:
        print("✓ Validation predictions generated for leaderboard")
    print("\nNext step: Run evaluate.py to generate detailed plots and analysis")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("Check training.log for details")
        raise   