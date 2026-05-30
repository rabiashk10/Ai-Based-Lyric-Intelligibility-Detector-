
# """
# =============================================================================
# FILE 2: run_experiments.py
# =============================================================================
# Run experiments to compare Baseline B vs Proposed Model

# This script:
# 1. Loads training data
# 2. Trains Baseline B model
# 3. Trains Proposed Enhanced model
# 4. Evaluates both on training set (since valid has no labels)
# 5. Saves models and results
# """

# import sys
# from pathlib import Path
# import logging
# import numpy as np
# import pandas as pd
# import sys
# from pathlib import Path

# ROOT = Path(__file__).resolve().parent.parent  # AiSemProject folder
# sys.path.append(str(ROOT))

# from data_loader import load_dataset_with_score


# # Add parent directory to path
# sys.path.insert(0, str(Path(__file__).resolve().parent))

# from data_loader import load_dataset_with_score
# from model_baseline_B import LogisticModelB
# from model_proposed import ProposedModelEnhanced

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('experiments.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Paths
# DATA_ROOT = r"C:\Users\Hina\AiSemProject\AI dataset"
# DATASET = "cadenza_data"
# SYSTEM = "stoi"
# SCORES_DIR = r"C:\Users\Hina\AiSemProject\clarity\recipes\cad_icassp_2026\baseline\precomputed"

# # Output directories
# MODELS_DIR = Path("models")
# RESULTS_DIR = Path("results")

# MODELS_DIR.mkdir(parents=True, exist_ok=True)
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# def load_train_data():
#     """Load training data correctly for Baseline B."""
#     logger.info("Loading training data...")
    
#     df = load_dataset_with_score(
#         data_root=DATA_ROOT,
#         dataset=DATASET,
#         split='train',
#         system=SYSTEM,
#         scores_dir=SCORES_DIR
#     )
    
#     # STOI feature as 2D array
#     X = df['stoi'].values.reshape(-1, 1)
    
#     # Keep correctness in 0-1 scale
#     y = df['correctness'].values
    
#     logger.info(f"Loaded {len(X)} training samples")
#     logger.info(f"STOI feature shape: {X.shape}, Correctness range: [{y.min():.4f}, {y.max():.4f}]")
    
#     return X, y


# def evaluate_model(model, X, y, model_name):
#     """Evaluate model and return metrics."""
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#     from scipy.stats import pearsonr
    
#     y_pred = model.predict(X)
    
#     rmse = np.sqrt(mean_squared_error(y, y_pred))
#     mae = mean_absolute_error(y, y_pred)
#     r2 = r2_score(y, y_pred)
#     corr, _ = pearsonr(y, y_pred)
    
#     metrics = {
#         'model': model_name,
#         'rmse': rmse,
#         'mae': mae,
#         'r2': r2,
#         'correlation': corr
#     }
    
#     logger.info(f"\n{model_name} Metrics:")
#     logger.info(f"  RMSE: {rmse:.4f}")
#     logger.info(f"  MAE: {mae:.4f}")
#     logger.info(f"  R²: {r2:.4f}")
#     logger.info(f"  Correlation: {corr:.4f}")
    
#     return metrics, y_pred


# def main():
#     logger.info("\n" + "="*70)
#     logger.info("ASSIGNMENT 3: RUNNING EXPERIMENTS")
#     logger.info("="*70)
    
#     # Load training data
#     X_train, y_train = load_train_data()
    
#     # ====================================================================
#     # STEP 1: Train Baseline B
#     # ====================================================================
#     logger.info("\n[STEP 1] Training Baseline B (Ridge Regression)...")
    
#     baseline_b = LogisticModelB(C=1.0, normalize=True)
#     baseline_b.fit(X_train, y_train)
#     baseline_b.save_model(str(MODELS_DIR / "baseline_B.pkl"))
    
#     baseline_metrics, baseline_pred = evaluate_model(
#         baseline_b, X_train, y_train, "Baseline B"
#     )
    
#     # ====================================================================
#     # STEP 2: Train Proposed Enhanced Model
#     # ====================================================================
#     logger.info("\n[STEP 2] Training Proposed Enhanced Model...")
    
#     proposed = ProposedModelEnhanced(poly_degree=2, cv_folds=5)
#     proposed.fit(X_train, y_train)
#     proposed.save_model(str(MODELS_DIR / "proposed_enhanced.pkl"))
    
#     proposed_metrics, proposed_pred = evaluate_model(
#         proposed, X_train, y_train, "Proposed Enhanced"
#     )
    
#     # ====================================================================
#     # STEP 3: Save Results
#     # ====================================================================
#     logger.info("\n[STEP 3] Saving Results...")
    
#     # Comparison DataFrame
#     comparison_df = pd.DataFrame([baseline_metrics, proposed_metrics])
#     comparison_csv = RESULTS_DIR / "model_comparison.csv"
#     comparison_df.to_csv(comparison_csv, index=False)
#     logger.info(f"Saved comparison: {comparison_csv}")
    
#     # Save predictions
#     predictions_df = pd.DataFrame({
#         'actual': y_train,
#         'baseline_pred': baseline_pred,
#         'proposed_pred': proposed_pred,
#         'baseline_error': np.abs(y_train - baseline_pred),
#         'proposed_error': np.abs(y_train - proposed_pred)
#     })
#     predictions_csv = RESULTS_DIR / "train_predictions.csv"
#     predictions_df.to_csv(predictions_csv, index=False)
#     logger.info(f"Saved predictions: {predictions_csv}")
    
#     # Save CV results
#     cv_results_df = pd.DataFrame(proposed.cv_results)
#     cv_results_csv = RESULTS_DIR / "proposed_cv_results.csv"
#     cv_results_df.to_csv(cv_results_csv, index=False)
#     logger.info(f"Saved CV results: {cv_results_csv}")
    
#     # ====================================================================
#     # FINAL SUMMARY
#     # ====================================================================
#     logger.info("\n" + "="*70)
#     logger.info("EXPERIMENTS COMPLETE!")
#     logger.info("="*70)
    
#     improvement = ((baseline_metrics['rmse'] - proposed_metrics['rmse']) / 
#                    baseline_metrics['rmse']) * 100
    
#     logger.info(f"\nTraining Set Performance:")
#     logger.info(f"  Baseline B: RMSE = {baseline_metrics['rmse']:.4f}, R² = {baseline_metrics['r2']:.4f}")
#     logger.info(f"  Proposed:   RMSE = {proposed_metrics['rmse']:.4f}, R² = {proposed_metrics['r2']:.4f}")
#     logger.info(f"\n  Improvement: {improvement:.2f}% reduction in RMSE")
    
#     logger.info(f"\nFiles saved in:")
#     logger.info(f"  Models: {MODELS_DIR}/")
#     logger.info(f"  Results: {RESULTS_DIR}/")
#     logger.info(f"\nNext: Run evaluate_results.py to generate plots")


# if __name__ == "__main__":
#     main()
# File: ASSIGN3/run_experiments.py
# File: ASSIGN3/run_experiments.py

import sys
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- NEW IMPORT for splitting data ---
from sklearn.model_selection import train_test_split 

# === Path setup ===
ROOT = Path(__file__).resolve().parent.parent  # -> AiSemProject
sys.path.append(str(ROOT))

try:
    import data_loader
except ImportError:
    logging.error("Failed to import data_loader. Make sure data_loader.py is in the root project directory.")
    sys.exit(1)

try:
    from model_baseline_B import LogisticModelB
except ImportError:
    logging.error("Failed to import model_baseline_B. Make sure model_baseline_B.py is in the root project directory.")
    sys.exit(1)

from ASSIGN3.model_proposed import ProposedEnhancedModel


# === Directory Setup ===
ASSIGN3_DIR = ROOT / "ASSIGN3"
RESULTS_DIR = ASSIGN3_DIR / "results"
PLOTS_DIR = ASSIGN3_DIR / "plots"
MODELS_DIR = ASSIGN3_DIR / "models"

for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Logging Setup ===
log_file = RESULTS_DIR / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Constants ===
DATA_ROOT_PATH = Path("C:/Users/Hina/AiSemProject/AI dataset")
DATASET = "cadenza_data"
SYSTEM = "stoi"
SCORES_DIR = "precomputed"
VALIDATION_SPLIT_SIZE = 0.2 # Use 20% for validation, just like your Phase 2
RANDOM_STATE = 42            # Ensures the split is the same every time


# ---------------------------------------------------------------------------
#                           HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def load_dataset_split(split: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load dataset split using existing data_loader.
    Returns the full DataFrame and the y labels.
    """
    logger.info(f"Loading {split} split...")
    
    df = data_loader.load_dataset_with_score(
        data_root=str(DATA_ROOT_PATH), # data_loader might expect string
        dataset=DATASET,
        split=split,
        system=SYSTEM,
        scores_dir=SCORES_DIR
    )
    
    logger.info(f"{split} set - Columns: {df.columns.tolist()}")
    logger.info(f"{split} set - Samples: {len(df)}")
    
    # We will return the full dataframe and select features in the main()
    logger.info(f"{split} set - All features loaded.")

    y = None
    if "correctness" in df.columns:
        y = df["correctness"].to_numpy()
        logger.info(f"{split} set - Correctness range: [{y.min():.4f}, {y.max():.4f}] (0-1 scale)")
    else:
        logger.warning(f"{split} set - No 'correctness' column found.")

    # --- FIX: Removed duplicated/dead code ---
    return df, y 


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    Handles 0-1 vs 0-100 scale conversion.
    """
    from scipy import stats
    
    # Ensure both are 0-100 scale for metrics
    if y_true.max() <= 1.0:
        y_true = y_true * 100.0
    if y_pred.max() <= 1.0:
        y_pred = y_pred * 100.0

    y_pred = np.clip(y_pred, 0, 100)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    pearson_r, pearson_p = stats.pearsonr(y_true.ravel(), y_pred.ravel())
    spearman_r, spearman_p = stats.spearmanr(y_true.ravel(), y_pred.ravel())
    
    return {
        f'{prefix}RMSE': rmse,
        f'{prefix}MAE': mae,
        f'{prefix}R2': r2,
        f'{prefix}Pearson_r': pearson_r,
        f'{prefix}Spearman_r': spearman_r,
    }


def train_baseline_b(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[LogisticModelB, Dict[str, Any]]:
    """
    Train Baseline B model.
    X_train is expected to be a 1D array (stoi only).
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING BASELINE B - Regularized Ridge Regression")
    logger.info("=" * 70)
    
    import time
    model = LogisticModelB(C=1.0, normalize=True)
    
    start_time = time.time()
    # .ravel() is correct here, as baseline model expects 1D input
    model.fit(X_train.ravel(), y_train) 
    train_time = time.time() - start_time
    
    y_train_pred = model.predict(X_train.ravel())
    
    train_metrics = compute_metrics(y_train, y_train_pred, prefix='train_')
    train_metrics['train_time'] = train_time
    
    logger.info(f"\nBaseline B Training Results (on {len(y_train)} samples):")
    logger.info(f"  Training Time: {train_time:.2f} seconds")
    logger.info(f"  Training RMSE: {train_metrics['train_RMSE']:.4f}")
    logger.info(f"  Training R²:   {train_metrics['train_R2']:.4f}")
    
    model_path = MODELS_DIR / 'baseline_b_model.pkl'
    model.save_model(str(model_path))
    
    return model, train_metrics


def train_proposed_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[ProposedEnhancedModel, Dict[str, Any]]:
    """
    Train Proposed Enhanced Model.
    X_train is expected to be a 2D array (N, 3 features).
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PROPOSED MODEL - Enhanced Non-Linear Ensemble")
    logger.info("=" * 70)
    
    import time
    
    model = ProposedEnhancedModel(
        poly_degree=2,  # --- FIX: Changed to 2 for a more robust model ---
        use_robust_scaling=True,
        ensemble_weights=(0.5, 0.3, 0.2),
        cv_folds=10,
        verbose=True
    )
    
    start_time = time.time()
    # --- CRITICAL FIX: Removed .ravel() ---
    # We are passing the (N, 3) feature array
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # --- CRITICAL FIX: Removed .ravel() ---
    y_train_pred = model.predict(X_train)

    train_metrics = compute_metrics(y_train, y_train_pred, prefix='train_')
    train_metrics['train_time'] = train_time
    
    logger.info(f"\nProposed Model Training Results (on {len(y_train)} samples):")
    logger.info(f"  Training Time: {train_time:.2f} seconds")
    logger.info(f"  Training RMSE: {train_metrics['train_RMSE']:.4f}")
    logger.info(f"  Training R²:   {train_metrics['train_R2']:.4f}")
    
    model_path = MODELS_DIR / 'proposed_model.pkl'
    model.save_model(str(model_path))
    
    return model, train_metrics


def evaluate_on_validation_set(
    baseline_model: LogisticModelB,
    proposed_model: ProposedEnhancedModel,
    X_valid_baseline: np.ndarray,  # --- FIX: New argument ---
    X_valid_proposed: np.ndarray,  # --- FIX: New argument ---
    y_valid: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate both models on the held-out validation set.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING MODELS ON HELD-OUT VALIDATION SET")
    logger.info("=" * 70)
    
    # --- CRITICAL FIX: Use the correct X for each model ---
    y_pred_baseline = baseline_model.predict(X_valid_baseline.ravel())
    y_pred_proposed = proposed_model.predict(X_valid_proposed)
    
    valid_metrics_baseline = compute_metrics(y_valid, y_pred_baseline, prefix='valid_')
    valid_metrics_proposed = compute_metrics(y_valid, y_pred_proposed, prefix='valid_')
    
    logger.info("\n" + "-" * 70)
    logger.info("BASELINE B VALIDATION RESULTS:")
    logger.info("-" * 70)
    for metric, value in valid_metrics_baseline.items():
        logger.info(f"  {metric:20s}: {value:.4f}")
    
    logger.info("\n" + "-" * 70)
    logger.info("PROPOSED MODEL VALIDATION RESULTS:")
    logger.info("-" * 70)
    for metric, value in valid_metrics_proposed.items():
        logger.info(f"  {metric:20s}: {value:.4f}")
    
    # ... (Rest of the function is the same) ...
    
    logger.info("\n" + "-" * 70)
    logger.info("IMPROVEMENTS (Proposed vs Baseline B):")
    logger.info("-" * 70)
    
    rmse_improvement = valid_metrics_baseline['valid_RMSE'] - valid_metrics_proposed['valid_RMSE']
    rmse_improvement_pct = (rmse_improvement / (valid_metrics_baseline['valid_RMSE'] + 1e-8)) * 100
    logger.info(f"  RMSE: {rmse_improvement:.4f} ({rmse_improvement_pct:+.2f}%)")
    
    return {
        'baseline': valid_metrics_baseline,
        'proposed': valid_metrics_proposed,
        'improvement': {
            'rmse_absolute': rmse_improvement,
            'rmse_relative_pct': rmse_improvement_pct,
        },
        'predictions': {
            'y_true': (y_valid * 100.0).tolist(),
            'y_pred_baseline': y_pred_baseline.tolist(),
            'y_pred_proposed': y_pred_proposed.tolist()
        }
    }


def generate_comparison_table(
    train_metrics_baseline: Dict,
    train_metrics_proposed: Dict,
    valid_results: Dict
) -> pd.DataFrame:
    """Generate comparison table for report."""
    
    metrics_to_compare = ['RMSE', 'MAE', 'R2', 'Pearson_r']
    rows = []
    for metric in metrics_to_compare:
        rows.append({
            'Metric': metric,
            'Baseline_B_Train': train_metrics_baseline.get(f'train_{metric}'),
            'Baseline_B_Valid': valid_results['baseline'].get(f'valid_{metric}'),
            'Proposed_Train': train_metrics_proposed.get(f'train_{metric}'),
            'Proposed_Valid': valid_results['proposed'].get(f'valid_{metric}'),
        })
    
    rows.append({
        'Metric': 'Training_Time_s',
        'Baseline_B_Train': train_metrics_baseline.get('train_time'),
        'Baseline_B_Valid': '-',
        'Proposed_Train': train_metrics_proposed.get('train_time'),
        'Proposed_Valid': '-',
    })
    
    df = pd.DataFrame(rows).round(4)
    return df


def plot_learning_curve(proposed_model: ProposedEnhancedModel):
    """Generate learning curve showing CV RMSE for different models."""
    logger.info("Generating learning curve (CV RMSE comparison)...")
    
    if not hasattr(proposed_model, 'cv_scores') or proposed_model.cv_scores is None:
        logger.warning("No CV scores available. Skipping learning curve.")
        return
    # ... (Plotting code from your script) ...
    # This will fail if cv_scores is not saved. Check your model_proposed.py
    logger.info(f"Saved learning curve to: {PLOTS_DIR / 'learning_curve_cv_rmse.pdf'}")


def generate_prediction_scatter(valid_results: Dict):
    """Generate prediction scatter plot comparing both models."""
    logger.info("Generating prediction scatter plot...")
    
    y_true = np.array(valid_results['predictions']['y_true'])
    y_pred_baseline = np.array(valid_results['predictions']['y_pred_baseline'])
    y_pred_proposed = np.array(valid_results['predictions']['y_pred_proposed'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    axes[0].scatter(y_true, y_pred_baseline, alpha=0.3, s=15, label='Predictions', color='#3498db')
    axes[0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    axes[0].set_xlabel('True Correctness (%)')
    axes[0].set_ylabel('Predicted Correctness (%)')
    axes[0].set_title(f"Baseline B (Valid RMSE: {valid_results['baseline']['valid_RMSE']:.2f})", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 100])
    axes[0].set_ylim([0, 100])
    
    axes[1].scatter(y_true, y_pred_proposed, alpha=0.3, s=15, color='#2ecc71', label='Predictions')
    axes[1].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    axes[1].set_xlabel('True Correctness (%)')
    axes[1].set_ylabel('')
    axes[1].set_title(f"Proposed Model (Valid RMSE: {valid_results['proposed']['valid_RMSE']:.2f})", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 100])
    axes[1].set_ylim([0, 100])
    
    fig.suptitle('Prediction Scatter Plot (Validation Set)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    pdf_path = PLOTS_DIR / 'prediction_scatter.pdf'
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved prediction scatter to: {pdf_path}")



# --- ADD THIS NEW FUNCTION TO YOUR run_experiments.py ---

def generate_submission_file(
    proposed_model: ProposedEnhancedModel, 
    data_root_path: Path, 
    results_dir: Path
):
    """
    Loads the *official* validation data (with no labels),
    prepares the features, makes predictions, and saves them
    in the 2-column, no-header format for challenge submission.
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: GENERATING OFFICIAL SUBMISSION FILE")
    logger.info("=" * 70)
    
    try:
        # 1. Load the official 'valid' data (no 'correctness' column)
        df_valid, _ = load_dataset_split('valid')
        
        # 2. Get the signal_ID
        if 'signal' not in df_valid.columns:
            logger.error("Submission file requires a 'signal' column for signal_ID.")
            return
        
        signal_ids = df_valid['signal'].values
        
        # 3. Prepare the 3 features (stoi, n_words, hearing_loss)
        features_to_use = ["stoi", "n_words", "hearing_loss"]
        df_valid_processed = df_valid[features_to_use].copy().fillna(0)

        # Use the *exact same mapping* as in training
        hl_mapping = {
            0: 0,
            'No Loss': 0,
            'Mild': 1,
            'Moderate': 2,
            'severe': 3 # Just in case
        }
        
        df_valid_proposed_features = df_valid_processed.copy()
        df_valid_proposed_features['hearing_loss'] = df_valid_processed['hearing_loss'].map(hl_mapping).fillna(0)
        
        X_valid_proposed = df_valid_proposed_features.to_numpy()
        logger.info(f"Loaded {len(X_valid_proposed)} samples from 'valid_metadata.json' for submission.")

        # 4. Make predictions with your best model
        logger.info("Making predictions with the trained Proposed Model...")
        final_predictions = proposed_model.predict(X_valid_proposed)
        
        # 5. Create the 2-column DataFrame
        submission_df = pd.DataFrame({
            'signal_ID': signal_ids,
            'intelligibility_score': final_predictions
        })
        
        # 6. Save to CSV in the required format
        submission_path = results_dir / "submission_proposed_model.csv"
        submission_df.to_csv(
            submission_path,
            index=False,
            header=False  # <-- As per the guidelines
        )
        
        logger.info(f"✓ Official 2-column submission file saved to: {submission_path}")

    except Exception as e:
        logger.error(f"Failed to generate submission file: {e}", exc_info=True)

# ---------------------------------------------------------------------------
#                           MAIN FUNCTION (UPDATED)
# ---------------------------------------------------------------------------

def main():
    """Main experiment pipeline."""
    
    logger.info("\n" + "=" * 70)
    logger.info("ICASSP CADENZA 2026 - LYRIC INTELLIGIBILITY CHALLENGE")
    logger.info("Phase 3: Proposed Solution Experiments")
    logger.info("=" * 70)
    
    # === STEP 1: LOAD FULL TRAINING DATA ===
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOADING *FULL* TRAINING DATA")
    logger.info("=" * 70)
    
    try:
        # --- FIX: Renamed variables ---
        df_full_train, y_full_train = load_dataset_split('train')
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return
        
    if y_full_train is None:
        logger.error("Training set must have 'correctness' labels!")
        return
        
    # === STEP 2: CREATE VALIDATION SPLIT (THE FIX!) ===
    logger.info("\n" + "=" * 70)
    logger.info(f"STEP 2: CREATING VALIDATION SPLIT ({VALIDATION_SPLIT_SIZE*100}%)")
    logger.info("=" * 70)
    
    # --- FIX: Split the DataFrame, not non-existent X ---
    df_train, df_valid, y_train, y_valid = train_test_split(
        df_full_train, y_full_train, 
        test_size=VALIDATION_SPLIT_SIZE, 
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Original data ({len(y_full_train)} samples) split into:")
    logger.info(f"  New Training Set:   {len(y_train)} samples")
    logger.info(f"  New Validation Set: {len(y_valid)} samples")
    
    valid_split_name = 'validation_split' # We created our own

    # === STEP 3: PREPARE FEATURE SETS ===
    # --- CRITICAL FIX: This block is updated to handle text features ---
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: PREPARING FEATURE SETS")
    logger.info("=" * 70)
    
    # --- Feature Set 1: For Baseline B (1 feature) ---
    X_train_baseline = df_train["stoi"].to_numpy()
    X_valid_baseline = df_valid["stoi"].to_numpy()
    logger.info(f"Baseline features: ['stoi'] (Shape: {X_train_baseline.shape})")

    # --- Feature Set 2: For Proposed Model (3 features) ---
    features_to_use = ["stoi", "n_words", "hearing_loss"]
    
    # Create copies to avoid errors
    df_train_processed = df_train[features_to_use].copy().fillna(0)
    df_valid_processed = df_valid[features_to_use].copy().fillna(0)

    # --- NEW: Convert 'hearing_loss' from text to numbers ---
    # First, let's see what values are in there
    unique_hl = df_train_processed['hearing_loss'].unique()
    logger.info(f"Found unique 'hearing_loss' values: {unique_hl}")

    # Create a mapping
    # We assume '0' (from fillna) and 'none' are the same (no hearing loss)
    # Create a mapping
    hl_mapping = {
        0: 0,        # For the .fillna(0) values
        'No Loss': 0,  # <-- FIX: Matches your data
        'Mild': 1,     # <-- FIX: Matches your data
        'Moderate': 2, # <-- FIX: Matches your data
        'severe': 3    # (Leaving this just in case)
    }
    # Apply the mapping
    df_train_proposed = df_train_processed.copy()
    df_valid_proposed = df_valid_processed.copy()
    df_train_proposed['hearing_loss'] = df_train_processed['hearing_loss'].map(hl_mapping).fillna(0) # fillna for any unmapped values
    df_valid_proposed['hearing_loss'] = df_valid_processed['hearing_loss'].map(hl_mapping).fillna(0)
    
    logger.info(f"Mapped 'hearing_loss' to numbers. New unique values: {df_train_proposed['hearing_loss'].unique()}")
    
    # Now, all data is numerical. We can safely convert to numpy.
    X_train_proposed = df_train_proposed.to_numpy()
    X_valid_proposed = df_valid_proposed.to_numpy()
    
    logger.info(f"Proposed features: {features_to_use} (Shape: {X_train_proposed.shape})")

    # === STEP 4: TRAIN MODELS ===
    logger.info("\n" + "=" * 70)
    logger.info(f"STEP 4: TRAINING MODELS ON {len(y_train)} SAMPLES")
    logger.info("=" * 70)
    
    # --- FIX: Pass correct X to each model ---
    baseline_model, train_metrics_baseline = train_baseline_b(X_train_baseline, y_train)
    proposed_model, train_metrics_proposed = train_proposed_model(X_train_proposed, y_train)

    # === STEP 5: EVALUATE MODELS ===
    logger.info("\n" + "=" * 70)
    logger.info(f"STEP 5: EVALUATING MODELS ON {len(y_valid)} SAMPLES")
    logger.info("=" * 70)
    
    # --- FIX: Pass correct X sets to evaluation ---
    valid_results = evaluate_on_validation_set(
        baseline_model, proposed_model,
        X_valid_baseline, X_valid_proposed,  # <-- Pass both feature sets
        y_valid
    )
        
    # === STEP 6: SAVE RESULTS AND PLOTS ===
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: SAVING ARTIFACTS")
    logger.info("=" * 70)
    
    # Save validation predictions
    pred_df = pd.DataFrame({
        'stoi': X_valid_baseline.ravel(), # Use baseline X for simplicity
        'y_true_scaled_100': valid_results['predictions']['y_true'],
        'y_pred_baseline': valid_results['predictions']['y_pred_baseline'],
        'y_pred_proposed': valid_results['predictions']['y_pred_proposed'],
    })
    pred_df['error_baseline'] = pred_df['y_true_scaled_100'] - pred_df['y_pred_baseline']
    pred_df['error_proposed'] = pred_df['y_true_scaled_100'] - pred_df['y_pred_proposed']
    
    pred_csv_path = RESULTS_DIR / f'{valid_split_name}_predictions.csv'
    pred_df.to_csv(pred_csv_path, index=False)
    logger.info(f"Saved validation predictions to: {pred_csv_path}")
    
    # Generate comparison table
    comparison_df = generate_comparison_table(
        train_metrics_baseline,
        train_metrics_proposed,
        valid_results
    )
    comp_csv_path = RESULTS_DIR / 'metrics_comparison.csv'
    comparison_df.to_csv(comp_csv_path, index=False)
    logger.info(f"\n--- Final Metrics Comparison Table ---")
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    logger.info(f"\nSaved comparison table to: {comp_csv_path}")
    
    # Generate plots
    # plot_learning_curve(proposed_model) # This may fail if cv_scores not saved
    generate_prediction_scatter(valid_results)
    
    # Save experiment summary
    experiment_summary = {
        'timestamp': datetime.now().isoformat(),
        'data_root': str(DATA_ROOT_PATH),
        'n_full_train': len(y_full_train),
        'n_train_split': len(y_train),
        'n_valid_split': len(y_valid),
        'baseline_train_metrics': train_metrics_baseline,
        'proposed_train_metrics': train_metrics_proposed,
        'validation_results': valid_results
    }
    # ... (rest of JSON saving logic is fine) ...
    summary_path = RESULTS_DIR / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        json.dump(experiment_summary, f, indent=2, cls=NpEncoder)
    logger.info(f"Experiment summary saved to: {summary_path}")
    
    # === FINAL SUMMARY ===
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("=" * 70)
    
    baseline_valid_rmse = valid_results['baseline']['valid_RMSE']
    proposed_valid_rmse = valid_results['proposed']['valid_RMSE']
    
    logger.info(f"Baseline B Valid RMSE: {baseline_valid_rmse:.4f}")
    logger.info(f"Proposed Valid RMSE:   {proposed_valid_rmse:.4f}")
    logger.info(f"Improvement:           {valid_results['improvement']['rmse_absolute']:.4f} "
                f"({valid_results['improvement']['rmse_relative_pct']:.2f}%)")
    
    logger.info(f"\nComparison to original 35.33 RMSE target:")
    logger.info(f"  Target:            < 30.0")
    logger.info(f"  Original Baseline: 35.33")
    logger.info(f"  Proposed Model:    {proposed_valid_rmse:.4f}")
    
    if proposed_valid_rmse < 30:
        logger.info("  🚀 TARGET ACHIEVED! Validation RMSE < 30!")
    elif proposed_valid_rmse < 35.33:
         logger.info("  ✓ IMPROVEMENT MADE! Validation RMSE is better than 35.33.")
    else:
        logger.info("  ⚠️ WARNING: Validation RMSE is worse than the original 35.33 baseline.")
        
    logger.info("=" * 70)
    # Now that the model is trained, generate the official submission file
    generate_submission_file(proposed_model, DATA_ROOT_PATH, RESULTS_DIR)
if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    main()