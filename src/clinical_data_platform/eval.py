#!/usr/bin/env python3
"""
Model evaluation with cross-validation and calibration analysis.

This module provides comprehensive model evaluation including:
- k-fold cross-validation with stratification
- Calibration analysis and reliability diagrams
- Performance metrics (accuracy, ROC-AUC, PR-AUC, Brier score, ECE)
- Bootstrap confidence intervals
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, 
    brier_score_loss, confusion_matrix, classification_report
)
from sklearn.datasets import make_classification

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with calibration analysis."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize evaluator.
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            ECE score
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy and confidence in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def cross_validate_model(self, X: np.ndarray, y: np.ndarray, model=None) -> Dict[str, List[float]]:
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Sklearn model (defaults to LogisticRegression)
            
        Returns:
            Dictionary of metric scores for each fold
        """
        if model is None:
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
        metrics = {
            'accuracy': [],
            'roc_auc': [],
            'pr_auc': [],
            'brier_score': [],
            'ece': []
        }
        
        # Store predictions for overall calibration plot
        all_y_true = []
        all_y_prob = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train, y_train)
            
            # Predictions
            y_pred_proba = model_fold.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_val, y_pred_proba > 0.5))
            metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            metrics['pr_auc'].append(average_precision_score(y_val, y_pred_proba))
            metrics['brier_score'].append(brier_score_loss(y_val, y_pred_proba))
            metrics['ece'].append(self.expected_calibration_error(y_val, y_pred_proba))
            
            # Store for overall calibration
            all_y_true.extend(y_val)
            all_y_prob.extend(y_pred_proba)
            
            logger.info(f"Fold {fold + 1}/{self.n_splits} completed")
            
        # Store overall predictions for calibration plot
        self.y_true_all = np.array(all_y_true)
        self.y_prob_all = np.array(all_y_prob)
        
        return metrics
    
    def bootstrap_confidence_intervals(self, scores: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals.
        
        Args:
            scores: List of metric scores
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Dictionary with mean, std, and CI bounds
        """
        scores = np.array(scores)
        n_scores = len(scores)
        
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(scores, size=n_scores, replace=True)
            bootstrap_scores.append(np.mean(bootstrap_sample))
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'ci_lower': float(np.percentile(bootstrap_scores, lower_percentile)),
            'ci_upper': float(np.percentile(bootstrap_scores, upper_percentile)),
            'n_folds': len(scores)
        }
    
    def generate_calibration_plot(self, output_path: Path) -> None:
        """
        Generate reliability diagram (calibration plot).
        
        Args:
            output_path: Path to save the plot
        """
        if not hasattr(self, 'y_true_all') or not hasattr(self, 'y_prob_all'):
            raise ValueError("Must run cross_validate_model first")
            
        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration plot
        fraction_positives, mean_predicted_value = calibration_curve(
            self.y_true_all, self.y_prob_all, n_bins=10
        )
        
        ax1.plot(mean_predicted_value, fraction_positives, "s-", label="Model", color='red', linewidth=2, markersize=8)
        ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.7)
        ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax1.set_ylabel("Fraction of Positives", fontsize=12)
        ax1.set_title("Reliability Diagram (Calibration Plot)", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Add ECE annotation
        ece_score = self.expected_calibration_error(self.y_true_all, self.y_prob_all)
        ax1.text(0.05, 0.95, f'ECE: {ece_score:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=11, fontweight='bold')
        
        # Histogram of predicted probabilities
        ax2.hist(self.y_prob_all[self.y_true_all == 0], bins=20, alpha=0.7, 
                label='Negative class', color='blue', density=True)
        ax2.hist(self.y_prob_all[self.y_true_all == 1], bins=20, alpha=0.7, 
                label='Positive class', color='red', density=True)
        ax2.set_xlabel("Predicted Probability", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title("Distribution of Predicted Probabilities", fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration plot saved to {output_path}")
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray, model=None, output_dir: Path = None) -> Dict[str, Any]:
        """
        Complete model evaluation pipeline.
        
        Args:
            X: Feature matrix
            y: Target vector  
            model: Sklearn model (defaults to LogisticRegression)
            output_dir: Directory to save artifacts
            
        Returns:
            Complete evaluation results
        """
        if output_dir is None:
            output_dir = Path("artifacts/eval")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting {self.n_splits}-fold cross-validation evaluation")
        
        # Run cross-validation
        cv_metrics = self.cross_validate_model(X, y, model)
        
        # Calculate confidence intervals
        results = {}
        for metric_name, scores in cv_metrics.items():
            results[metric_name] = self.bootstrap_confidence_intervals(scores)
            
        # Add dataset info
        results['dataset_info'] = {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_positive': int(np.sum(y)),
            'n_negative': int(np.sum(1 - y)),
            'positive_rate': float(np.mean(y))
        }
        
        results['evaluation_config'] = {
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'model_type': model.__class__.__name__ if model else 'LogisticRegression'
        }
        
        # Save metrics to JSON
        metrics_path = output_dir / "cv_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"CV metrics saved to {metrics_path}")
        
        # Generate calibration plot
        calibration_path = output_dir / "calibration.png"
        self.generate_calibration_plot(calibration_path)
        
        # Print summary
        self.print_evaluation_summary(results)
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted evaluation summary."""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        dataset_info = results['dataset_info']
        print(f"Dataset: {dataset_info['n_samples']} samples, {dataset_info['n_features']} features")
        print(f"Class balance: {dataset_info['positive_rate']:.1%} positive")
        print(f"Cross-validation: {results['evaluation_config']['n_splits']} folds")
        
        print(f"\n{'Metric':<15} {'Mean':<8} {'±95% CI':<15} {'Std':<8}")
        print("-" * 50)
        
        for metric in ['accuracy', 'roc_auc', 'pr_auc', 'brier_score', 'ece']:
            if metric in results:
                stats = results[metric]
                ci_width = stats['ci_upper'] - stats['ci_lower']
                print(f"{metric:<15} {stats['mean']:.3f}    ±{ci_width/2:.3f}         {stats['std']:.3f}")
        
        print("="*60)


def generate_synthetic_clinical_data(n_samples: int = 1000, n_features: int = 20, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic clinical data for evaluation.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        Feature matrix and target vector
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.1),
        n_clusters_per_class=2,
        weights=[0.7, 0.3],  # Imbalanced classes
        flip_y=0.05,  # 5% label noise
        random_state=random_state
    )
    
    return X, y


def main():
    """Main evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model evaluation with cross-validation and calibration")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval", help="Output directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "rf"], 
                       help="Model type")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Generate data
    logger.info("Generating synthetic clinical data...")
    X, y = generate_synthetic_clinical_data()
    
    # Select model
    if args.model == "logistic":
        model = LogisticRegression(random_state=args.random_state, max_iter=1000)
    else:
        model = RandomForestClassifier(random_state=args.random_state, n_estimators=100)
    
    # Run evaluation
    evaluator = ModelEvaluator(n_splits=args.n_folds, random_state=args.random_state)
    results = evaluator.evaluate_model(X, y, model, Path(args.output_dir))
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()