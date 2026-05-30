"""
Data loader module for Cadenza CLIP1 Challenge
Loads precomputed STOI scores and metadata for intelligibility prediction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def read_jsonl(filepath: str) -> List[Dict]:
    """
    Read a JSONL file and return list of dictionaries.
    
    Args:
        filepath (str): Path to the JSONL file
        
    Returns:
        List[Dict]: List of records from JSONL file
    """
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} records from {filepath}")
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON in {filepath}: {e}")
        raise
    
    return records


def load_metadata(data_root: str, dataset: str, split: str) -> List[Dict]:
    """
    Load metadata JSON file containing ground truth correctness values.
    
    Args:
        data_root (str): Root directory containing cadenza_data
        dataset (str): Dataset name (e.g., 'cadenza_data')
        split (str): Data split ('train' or 'valid')
        
    Returns:
        List[Dict]: List of metadata records
    """
    metadata_path = Path(data_root) / dataset / "metadata" / f"{split}_metadata.json"
    
    logger.info(f"Loading metadata from: {metadata_path}")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        logger.info(f"Loaded {len(records)} metadata records for {split} split")
        return records
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata JSON: {e}")
        raise


def load_precomputed_scores(
    dataset: str, 
    split: str, 
    system: str = 'stoi',
    scores_dir: str = 'precomputed'
) -> Dict[str, float]:
    """
    Load precomputed STOI or Whisper scores from JSONL file.
    
    Args:
        dataset (str): Dataset name (e.g., 'cadenza_data')
        split (str): Data split ('train' or 'valid')
        system (str): Scoring system ('stoi' or 'whisper')
        scores_dir (str): Directory containing score files (default: 'precomputed')
        
    Returns:
        Dict[str, float]: Dictionary mapping signal names to scores
    """
    # Construct the score file path
    score_filename = f"{dataset}.{split}.{system}.jsonl"
    score_path = Path(scores_dir) / score_filename
    
    logger.info(f"Loading precomputed {system.upper()} scores from: {score_path}")
    
    try:
        scores = read_jsonl(str(score_path))
        
        # Create index for fast lookup: {signal_name: score}
        score_index = {
            record["signal"]: record[system] 
            for record in scores
        }
        
        logger.info(f"Loaded {len(score_index)} {system.upper()} scores")
        return score_index
        
    except FileNotFoundError:
        logger.error(f"Score file not found: {score_path}")
        logger.info(f"Make sure you have precomputed scores in {scores_dir}/")
        raise


def load_dataset_with_score(
    data_root: str,
    dataset: str,
    split: str,
    system: str = 'stoi',
    scores_dir: str = 'precomputed'
) -> pd.DataFrame:
    """
    Load dataset with both metadata and precomputed scores merged.
    
    This is the main function to load data for training/evaluation.
    
    Args:
        data_root (str): Root directory containing cadenza_data
        dataset (str): Dataset name (e.g., 'cadenza_data')
        split (str): Data split ('train' or 'valid')
        system (str): Scoring system ('stoi' or 'whisper')
        scores_dir (str): Directory containing score files
        
    Returns:
        pd.DataFrame: DataFrame with columns [signal, correctness, stoi/whisper, ...]
    """
    logger.info(f"Loading {split} dataset with {system.upper()} scores...")
    
    # Load metadata (ground truth)
    metadata_records = load_metadata(data_root, dataset, split)
    
    # Load precomputed scores
    score_index = load_precomputed_scores(dataset, split, system, scores_dir)
    
    # Merge scores into metadata
    for record in metadata_records:
        signal_name = record["signal"]
        if signal_name in score_index:
            record[system] = score_index[signal_name]
        else:
            logger.warning(f"No {system} score found for signal: {signal_name}")
            record[system] = None
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata_records)
    
    # Remove rows with missing scores
    initial_count = len(df)
    df = df.dropna(subset=[system])
    final_count = len(df)
    
    if initial_count != final_count:
        logger.warning(f"Dropped {initial_count - final_count} rows with missing scores")
    
    logger.info(f"Loaded dataset: {len(df)} samples with {system.upper()} scores")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def get_dataset_statistics(df: pd.DataFrame, system: str = 'stoi') -> Dict:
    """
    Compute basic statistics for the loaded dataset.
    
    Args:
        df (pd.DataFrame): Dataset DataFrame
        system (str): Scoring system column name
        
    Returns:
        Dict: Dictionary containing dataset statistics
    """
    stats = {
        'num_samples': len(df),
        'score_mean': df[system].mean(),
        'score_std': df[system].std(),
        'score_min': df[system].min(),
        'score_max': df[system].max(),
        'correctness_mean': df['correctness'].mean(),
        'correctness_std': df['correctness'].std(),
        'correctness_min': df['correctness'].min(),
        'correctness_max': df['correctness'].max(),
    }
    
    return stats


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration (update these paths for your setup)
    DATA_ROOT = "C:/Code/AISEMPROJECT"  # Parent folder of cadenza_data
    DATASET = "cadenza_data"
    SPLIT = "train"
    SYSTEM = "stoi"
    
    try:
        # Load dataset
        df = load_dataset_with_score(
            data_root=DATA_ROOT,
            dataset=DATASET,
            split=SPLIT,
            system=SYSTEM,
            scores_dir='precomputed'  # Using precomputed folder
        )
        
        # Display sample
        print("\n=== Dataset Sample ===")
        print(df.head())
        
        # Display statistics
        print("\n=== Dataset Statistics ===")
        stats = get_dataset_statistics(df, SYSTEM)
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
        
        # Display data types
        print("\n=== Data Types ===")
        print(df.dtypes)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise