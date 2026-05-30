"""
Evaluation Script for Cadenza CLIP1 Challenge
Evaluates trained baseline models on train set and generates predictions for validation set.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

from data_loader import load_dataset_with_score
from preprocess import prepare_training_data
from model_baseline_A import LogisticModelA
from model_baseline_B import LogisticModelB
from model_baseline_C import PolynomialModelC

# ---------------- CONFIGURATION ---------------- #
class EvaluationConfig:
    DATA_ROOT = "C:/Code/AISEMPROJECT"
    DATASET = "cadenza_data"
    SYSTEM = "stoi"
    SCORES_DIR = "precomputed"
    
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    PLOTS_DIR = "plots"
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    FIGURE_DPI = 300
    FIGURE_SIZE = (10, 6)

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('evaluation.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
sns.set_style("darkgrid")


# ---------------- HELPERS ---------------- #
def setup_directories():
    Path(EvaluationConfig.RESULTS_DIR).mkdir(exist_ok=True)
    Path(EvaluationConfig.PLOTS_DIR).mkdir(exist_ok=True)
    logger.info(f"Directories ready: {EvaluationConfig.RESULTS_DIR}, {EvaluationConfig.PLOTS_DIR}")


def load_trained_models():
    models = {}
    for name, cls, fname in [
        ('baseline_A', LogisticModelA, 'baseline_A.pkl'),
        ('baseline_B', LogisticModelB, 'baseline_B.pkl'),
        ('baseline_C', PolynomialModelC, 'baseline_C.pkl')
    ]:
        path = Path(EvaluationConfig.MODELS_DIR) / fname
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        model = cls.load_model(str(path))
        models[name] = model
        logger.info(f"Loaded {name}")
    return models


def compute_metrics(preds, labels):
    # Convert to 0-100 scale if needed
    if labels.max() <= 1.0:
        labels = labels * 100
    if preds.max() <= 1.0:
        preds = preds * 100

    errors = preds - labels
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    rmse = np.sqrt(np.mean(squared_errors))
    mae = np.mean(abs_errors)
    max_error = np.max(abs_errors)
    ncc, _ = pearsonr(preds, labels)
    scc, _ = spearmanr(preds, labels)
    kt, _ = kendalltau(preds, labels)
    r2 = 1 - (np.sum(squared_errors)/np.sum((labels - np.mean(labels))**2))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'Max_Error': max_error,
        'NCC': ncc,
        'SCC': scc,
        'KT': kt,
        'R2': r2,
        'Error_Mean': np.mean(errors),
        'Error_Std': np.std(errors),
        'MAE_25': np.percentile(abs_errors, 25),
        'MAE_50': np.percentile(abs_errors, 50),
        'MAE_75': np.percentile(abs_errors, 75),
        'n_samples': len(preds)
    }


def plot_predictions_vs_truth(pred_dict, y, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    y = y * 100 if y.max() <= 1.0 else y
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    names = ['Baseline A', 'Baseline B', 'Baseline C']

    for i, (k, name, color) in enumerate(zip(['baseline_A','baseline_B','baseline_C'], names, colors)):
        ax = axes[i]
        p = pred_dict[k]
        ax.scatter(y, p, alpha=0.4, s=20, color=color)
        min_val = min(y.min(), p.min())
        max_val = max(y.max(), p.max())
        ax.plot([min_val,max_val],[min_val,max_val],'k--', linewidth=2)
        metrics = compute_metrics(p,y)
        ax.set_title(f"{name}\nRMSE: {metrics['RMSE']:.2f}, NCC: {metrics['NCC']:.3f}")
        ax.set_xlabel("Ground Truth (%)")
        ax.set_ylabel("Predicted (%)")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=EvaluationConfig.FIGURE_DPI)
    plt.close()
    logger.info(f"Saved plot: {save_path}")


def plot_model_comparison(metrics_dict, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    models = list(metrics_dict.keys())
    rmse_vals = [metrics_dict[m]['RMSE'] for m in models]
    ax.bar(models, rmse_vals, color=['#FF6B6B','#4ECDC4','#45B7D1'])
    ax.set_ylabel("RMSE")
    ax.set_title("Train RMSE Comparison")
    plt.savefig(save_path, dpi=EvaluationConfig.FIGURE_DPI)
    plt.close()
    logger.info(f"Saved model comparison plot: {save_path}")


# ---------------- MAIN PIPELINE ---------------- #
def main():
    setup_directories()
    models = load_trained_models()

    # ---------------- TRAIN EVALUATION ---------------- #
    logger.info("Loading train set...")
    train_df = load_dataset_with_score(
        EvaluationConfig.DATA_ROOT, EvaluationConfig.DATASET, split='train',
        system=EvaluationConfig.SYSTEM, scores_dir=EvaluationConfig.SCORES_DIR
    )

    data = prepare_training_data(train_df, test_size=EvaluationConfig.TEST_SIZE,
                                 random_state=EvaluationConfig.RANDOM_STATE,
                                 normalize=False)
    X_test = data['X_test']
    y_test = data['y_test']

    predictions_dict = {}
    metrics_dict = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        predictions_dict[name] = preds
        metrics_dict[name] = compute_metrics(preds, y_test)
        logger.info(f"{name}: RMSE={metrics_dict[name]['RMSE']:.4f}, NCC={metrics_dict[name]['NCC']:.4f}")

    # Generate plots
    plot_predictions_vs_truth(predictions_dict, y_test,
                              save_path=Path(EvaluationConfig.PLOTS_DIR)/"predictions_vs_train_truth.pdf")
    plot_model_comparison(metrics_dict,
                          save_path=Path(EvaluationConfig.PLOTS_DIR)/"train_model_comparison.pdf")

    # Save metrics JSON
    with open(Path(EvaluationConfig.RESULTS_DIR)/"train_metrics.json",'w') as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info("Saved train metrics")

    # ---------------- VALIDATION PREDICTIONS ---------------- #
    logger.info("Loading validation set...")
    valid_df = load_dataset_with_score(
        EvaluationConfig.DATA_ROOT, EvaluationConfig.DATASET, split='valid',
        system=EvaluationConfig.SYSTEM, scores_dir=EvaluationConfig.SCORES_DIR
    )
    X_valid = valid_df.filter(like='stoi').to_numpy().flatten()  # <-- Flatten to 1D for model input

    for name, model in models.items():
        preds = model.predict(X_valid)
        df_out = pd.DataFrame({
            'id': valid_df['id'] if 'id' in valid_df.columns else np.arange(len(valid_df)),
            'predicted_score': preds
        })
        df_out.to_csv(Path(EvaluationConfig.RESULTS_DIR)/f"{name}_validation_predictions.csv", index=False)
        logger.info(f"Saved {name} validation predictions CSV")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
