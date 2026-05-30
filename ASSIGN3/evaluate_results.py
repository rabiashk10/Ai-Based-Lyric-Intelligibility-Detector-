# File: ASSIGN3/evaluate_results.py

import sys
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Path setup ===
# Assumes this script is in ASSIGN3/
ROOT = Path(__file__).resolve().parent.parent  # -> AiSemProject
sys.path.append(str(ROOT))

# Import your model classes (needed to load the .pkl files)
try:
    from model_baseline_B import LogisticModelB
    from ASSIGN3.model_proposed import ProposedEnhancedModel
except ImportError:
    logging.warning("Could not import model classes. Feature importance plot may fail.")

# === Directory Setup ===
ASSIGN3_DIR = ROOT / "ASSIGN3"
RESULTS_DIR = ASSIGN3_DIR / "results"
PLOTS_DIR = ASSIGN3_DIR / "plots"
MODELS_DIR = ASSIGN3_DIR / "models"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def plot_comparison_bar(comparison_df: pd.DataFrame):
    """
    Plot 1: Comparison bar chart for key metrics (RMSE, MAE, R²).
    """
    logger.info("Generating Metric Comparison Bar Chart...")
    
    try:
        # Filter for the main metrics
        metrics_to_plot = ['RMSE', 'MAE', 'R2']
        plot_df = comparison_df[comparison_df['Metric'].isin(metrics_to_plot)]
        
        # Melt the DataFrame for easy plotting with Seaborn
        plot_df = plot_df.melt(
            id_vars='Metric',
            value_vars=['Baseline_B_Valid', 'Proposed_Valid'],
            var_name='Model',
            value_name='Score'
        )
        
        # Clean up model names
        plot_df['Model'] = plot_df['Model'].replace({
            'Baseline_B_Valid': 'Baseline B (1-Feature)',
            'Proposed_Valid': 'Proposed (3-Features)'
        })

        # Create a 1x3 subplot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Comparison (Validation Set)', fontsize=16, fontweight='bold')
        
        # Plot RMSE
        sns.barplot(
            x='Model', y='Score', data=plot_df[plot_df['Metric'] == 'RMSE'],
            ax=axes[0], palette=['#e74c3c', '#2ecc71']
        )
        axes[0].set_title('Root Mean Squared Error (RMSE)', fontsize=13)
        axes[0].set_ylabel('RMSE (Lower is better)')
        axes[0].set_xlabel(None)
        
        # Plot MAE
        sns.barplot(
            x='Model', y='Score', data=plot_df[plot_df['Metric'] == 'MAE'],
            ax=axes[1], palette=['#e74c3c', '#2ecc71']
        )
        axes[1].set_title('Mean Absolute Error (MAE)', fontsize=13)
        axes[1].set_ylabel('MAE (Lower is better)')
        axes[1].set_xlabel(None)

        # Plot R2
        sns.barplot(
            x='Model', y='Score', data=plot_df[plot_df['Metric'] == 'R2'],
            ax=axes[2], palette=['#e74c3c', '#2ecc71']
        )
        axes[2].set_title('R² Score', fontsize=13)
        axes[2].set_ylabel('R² (Higher is better)')
        axes[2].set_xlabel(None)

        # Add value labels
        for ax in axes:
            for p in ax.patches:
                ax.annotate(
                    f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, 
                    color='black', xytext=(0, 5), textcoords='offset points'
                )
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = PLOTS_DIR / "1_comparison_bar_chart.pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_path.name}")

    except Exception as e:
        logger.error(f"Could not generate comparison bar chart: {e}")


def plot_scatter_plots(predictions_df: pd.DataFrame):
    """
    Plot 2: Predicted vs. Actual scatter plots for both models.
    """
    logger.info("Generating Prediction Scatter Plots...")
    
    try:
        y_true = predictions_df['y_true_scaled_100']
        y_pred_baseline = predictions_df['y_pred_baseline']
        y_pred_proposed = predictions_df['y_pred_proposed']

        # Get R² scores
        from sklearn.metrics import r2_score
        r2_baseline = r2_score(y_true, y_pred_baseline)
        r2_proposed = r2_score(y_true, y_pred_proposed)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        fig.suptitle('Predicted vs. Actual Correctness (Validation Set)', fontsize=16, fontweight='bold')

        # Baseline
        axes[0].scatter(y_true, y_pred_baseline, alpha=0.3, s=15, 
                        label='Predictions', color='#e74c3c', edgecolors='k', linewidth=0.2)
        axes[0].plot([0, 100], [0, 100], 'k--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Correctness (%)', fontsize=12)
        axes[0].set_ylabel('Predicted Correctness (%)', fontsize=12)
        axes[0].set_title(f'Baseline B (R² = {r2_baseline:.4f})', fontsize=14)
        axes[0].legend()
        axes[0].set_xlim([0, 100])
        axes[0].set_ylim([0, 100])

        # Proposed
        axes[1].scatter(y_true, y_pred_proposed, alpha=0.3, s=15, 
                        label='Predictions', color='#2ecc71', edgecolors='k', linewidth=0.2)
        axes[1].plot([0, 100], [0, 100], 'k--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Correctness (%)', fontsize=12)
        axes[1].set_ylabel('')
        axes[1].set_title(f'Proposed Model (R² = {r2_proposed:.4f})', fontsize=14)
        axes[1].legend()
        axes[1].set_xlim([0, 100])
        axes[1].set_ylim([0, 100])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = PLOTS_DIR / "2_scatter_plots.pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_path.name}")
        
    except Exception as e:
        logger.error(f"Could not generate scatter plots: {e}")


def plot_residual_plots(predictions_df: pd.DataFrame):
    """
    Plot 3: Residual plots for both models.
    """
    logger.info("Generating Residual Plots...")
    
    try:
        y_pred_baseline = predictions_df['y_pred_baseline']
        y_pred_proposed = predictions_df['y_pred_proposed']
        error_baseline = predictions_df['error_baseline']
        error_proposed = predictions_df['error_proposed']

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        fig.suptitle('Residuals (Predicted - Actual)', fontsize=16, fontweight='bold')

        # Baseline
        sns.scatterplot(
            x=y_pred_baseline, y=error_baseline, ax=axes[0], 
            alpha=0.3, s=15, color='#e74c3c', edgecolors='k', linewidth=0.2
        )
        axes[0].axhline(y=0, color='k', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Correctness (%)', fontsize=12)
        axes[0].set_ylabel('Residual (Error)', fontsize=12)
        axes[0].set_title('Baseline B', fontsize=14)

        # Proposed
        sns.scatterplot(
            x=y_pred_proposed, y=error_proposed, ax=axes[1],
            alpha=0.3, s=15, color='#2ecc71', edgecolors='k', linewidth=0.2
        )
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Correctness (%)', fontsize=12)
        axes[1].set_ylabel('')
        axes[1].set_title('Proposed Model', fontsize=14)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = PLOTS_DIR / "3_residual_plots.pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_path.name}")

    except Exception as e:
        logger.error(f"Could not generate residual plots: {e}")


def plot_error_distributions(predictions_df: pd.DataFrame):
    """
    Plot 4: Error distribution histograms.
    """
    logger.info("Generating Error Distribution Histograms...")
    
    try:
        error_baseline = predictions_df['error_baseline']
        error_proposed = predictions_df['error_proposed']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle('Error Distribution', fontsize=16, fontweight='bold')

        # Baseline
        sns.histplot(error_baseline, bins=30, kde=True, ax=axes[0], color='#e74c3c')
        axes[0].axvline(error_baseline.mean(), color='red', linestyle='--', linewidth=2, 
                        label=f"Mean = {error_baseline.mean():.2f}\nStd = {error_baseline.std():.2f}")
        axes[0].set_xlabel('Error (Predicted - Actual)', fontsize=12)
        axes[0].set_title('Baseline B', fontsize=14)
        axes[0].legend()

        # Proposed
        sns.histplot(error_proposed, bins=30, kde=True, ax=axes[1], color='#2ecc71')
        axes[1].axvline(error_proposed.mean(), color='green', linestyle='--', linewidth=2, 
                        label=f"Mean = {error_proposed.mean():.2f}\nStd = {error_proposed.std():.2f}")
        axes[1].set_xlabel('Error (Predicted - Actual)', fontsize=12)
        axes[1].set_title('Proposed Model', fontsize=14)
        axes[1].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = PLOTS_DIR / "4_error_distributions.pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_path.name}")
        
    except Exception as e:
        logger.error(f"Could not generate error distributions: {e}")


def plot_feature_importance():
    """
    Plot 5: Feature importance plot from the proposed model.
    """
    logger.info("Generating Feature Importance Plot...")
    
    try:
        # Load the saved model
        model_path = MODELS_DIR / "proposed_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Get the internal PolynomialFeatures object
        poly = model.poly_features
        
        # Get the original feature names
        original_features = ['stoi', 'n_words', 'hearing_loss']
        
        # Get the new polynomial feature names
        # '1' is the bias term, which we skip
        poly_feature_names = poly.get_feature_names_out(original_features)
        
        # Get the coefficients from one of the internal models (e.g., Ridge)
        # We skip the first coefficient, which corresponds to the '1' bias term
        # if include_bias=True, but yours is False. Let's check.
        # Your poly_features was init with include_bias=False
        # But fit_transform *still* adds a bias column if interaction_only=False
        
        # Let's get the coefficients from the Ridge model
        coeffs = model.ridge_model.coef_
        
        # Match coefficients to names
        if len(coeffs) != len(poly_feature_names):
             logger.warning(f"Coeff count ({len(coeffs)}) mismatch with feature names ({len(poly_feature_names)}). Skipping importance plot.")
             # This can happen if the poly object wasn't saved correctly.
             # Let's try to get names from the number of features
             poly_feature_names = model.poly_features.get_feature_names_out(original_features)


        # Create a DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': poly_feature_names,
            'Importance (Abs Coeff)': np.abs(coeffs)
        }).sort_values(by='Importance (Abs Coeff)', ascending=False)
        
        plt.figure(figsize=(10, 7))
        sns.barplot(
            x='Importance (Abs Coeff)', y='Feature', 
            data=importance_df, palette='viridis'
        )
        plt.title('Feature Importance for Proposed Model (Ridge Coefficients)', fontsize=15, fontweight='bold')
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        save_path = PLOTS_DIR / "5_feature_importance.pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_path.name}")

    except Exception as e:
        logger.error(f"Could not generate feature importance plot: {e}")



def main():
    """
    Main function to load results and generate all plots.
    """
    logger.info("\n" + "="*70)
    logger.info("EVALUATING RESULTS AND GENERATING PLOTS FOR ASSIGNMENT 3")
    logger.info("="*70)
    
    # Define file paths
    comparison_file = RESULTS_DIR / "metrics_comparison.csv"
    predictions_file = RESULTS_DIR / "validation_split_predictions.csv"
    
    # Check if files exist
    if not comparison_file.exists():
        logger.error(f"FATAL: {comparison_file} not found.")
        logger.error("Please run 'run_experiments.py' first.")
        return
        
    if not predictions_file.exists():
        logger.error(f"FATAL: {predictions_file} not found.")
        logger.error("Please run 'run_experiments.py' first.")
        return

    # Load the results
    logger.info(f"Loading results from {RESULTS_DIR}...")
    comparison_df = pd.read_csv(comparison_file)
    predictions_df = pd.read_csv(predictions_file)
    
    # Generate all plots
    plot_comparison_bar(comparison_df)
    plot_scatter_plots(predictions_df)
    plot_residual_plots(predictions_df)
    plot_error_distributions(predictions_df)
    plot_feature_importance() # This is the high-value plot
    
    logger.info("\n" + "="*70)
    logger.info(f"✓ All plots saved to: {PLOTS_DIR}")
    logger.info("Evaluation complete.")
    logger.info("="*70)


if __name__ == "__main__":
    main()