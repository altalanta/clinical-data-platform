# Model Evaluation

This page documents our comprehensive model evaluation methodology including cross-validation, calibration analysis, and performance metrics.

## Evaluation Methodology

Our evaluation pipeline implements rigorous statistical validation:

- **5-fold stratified cross-validation** for robust performance estimation
- **Bootstrap confidence intervals** for statistical significance testing  
- **Calibration analysis** with reliability diagrams
- **Multiple performance metrics** including calibration quality

## Performance Metrics

### Classification Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | Fraction of correct predictions | Higher is better |
| **ROC-AUC** | Area under receiver operating curve | 0.5 = random, 1.0 = perfect |
| **PR-AUC** | Area under precision-recall curve | Better for imbalanced datasets |
| **Brier Score** | Mean squared difference between predicted probabilities and outcomes | Lower is better (0 = perfect) |
| **ECE** | Expected Calibration Error | Lower is better (0 = perfect calibration) |

### Latest Evaluation Results

```json
{
  "accuracy": {
    "mean": 0.852,
    "ci_lower": 0.831,
    "ci_upper": 0.873,
    "std": 0.014,
    "n_folds": 5
  },
  "roc_auc": {
    "mean": 0.918,
    "ci_lower": 0.897,
    "ci_upper": 0.939,
    "std": 0.012,
    "n_folds": 5
  },
  "pr_auc": {
    "mean": 0.734,
    "ci_lower": 0.701,
    "ci_upper": 0.767,
    "std": 0.019,
    "n_folds": 5
  },
  "brier_score": {
    "mean": 0.118,
    "ci_lower": 0.105,
    "ci_upper": 0.131,
    "std": 0.008,
    "n_folds": 5
  },
  "ece": {
    "mean": 0.034,
    "ci_lower": 0.021,
    "ci_upper": 0.047,
    "std": 0.008,
    "n_folds": 5
  }
}
```

## Calibration Analysis

Model calibration is critical for clinical applications where predicted probabilities must be reliable for decision-making.

![Calibration Plot](../artifacts/eval/calibration.png)

### Reliability Diagram

The calibration plot shows:

- **Left panel**: Reliability diagram comparing predicted vs actual probabilities
- **Right panel**: Distribution of predicted probabilities by class
- **ECE Score**: Expected Calibration Error quantifying calibration quality

### Calibration Quality Assessment

- **Well-calibrated model**: Points lie close to the diagonal line
- **Overconfident model**: Points lie below the diagonal  
- **Underconfident model**: Points lie above the diagonal
- **ECE < 0.05**: Generally considered well-calibrated

## Cross-Validation Results

### Performance Summary

| Metric | Mean ± 95% CI | Standard Deviation |
|--------|---------------|-------------------|
| Accuracy | 0.852 ± 0.021 | 0.014 |
| ROC-AUC | 0.918 ± 0.021 | 0.012 |
| PR-AUC | 0.734 ± 0.033 | 0.019 |
| Brier Score | 0.118 ± 0.013 | 0.008 |
| ECE | 0.034 ± 0.013 | 0.008 |

### Statistical Significance

All confidence intervals are computed using bootstrap resampling (1000 iterations) to ensure robust statistical inference.

## Dataset Information

- **Samples**: 1,000 clinical records
- **Features**: 20 clinical variables
- **Class Balance**: 30% positive cases
- **Data Quality**: 5% label noise (realistic clinical setting)

## Model Configuration

```python
# Current evaluation setup
evaluator = ModelEvaluator(
    n_splits=5,           # 5-fold cross-validation
    random_state=42       # Reproducible results
)

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  # Handle class imbalance
)
```

## Reproducing Results

Run the complete evaluation pipeline:

```bash
# Generate synthetic data and run evaluation
make demo-public && make eval

# View results
ls artifacts/eval/
# cv_metrics.json  calibration.png
```

### Alternative Models

Compare different model architectures:

```bash
# Logistic Regression (default)
python -m clinical_data_platform.eval --model logistic

# Random Forest  
python -m clinical_data_platform.eval --model rf

# Custom cross-validation folds
python -m clinical_data_platform.eval --n-folds 10
```

## Clinical Decision Thresholds

### Risk Stratification

Based on calibration analysis, recommended decision thresholds:

| Risk Level | Probability Threshold | Clinical Action |
|------------|----------------------|-----------------|
| **Low Risk** | < 0.3 | Standard monitoring |
| **Medium Risk** | 0.3 - 0.7 | Enhanced monitoring |
| **High Risk** | > 0.7 | Immediate intervention |

### Threshold Selection Criteria

1. **Sensitivity vs Specificity**: Balance false positives/negatives
2. **Clinical Cost**: Consider intervention costs and risks
3. **Calibration Quality**: Ensure reliable probability estimates
4. **Population Prevalence**: Adjust for local patient populations

## Limitations and Considerations

### Statistical Limitations

- **Sample Size**: 1,000 samples may limit generalizability
- **Synthetic Data**: Real clinical data may show different patterns
- **Feature Engineering**: Limited to basic statistical features

### Clinical Limitations  

- **External Validation**: Requires validation on independent datasets
- **Temporal Stability**: Model performance may drift over time
- **Population Bias**: Training data may not represent all patient groups
- **Regulatory Approval**: Clinical deployment requires regulatory validation

## Monitoring and Maintenance

### Production Monitoring

- **Performance Drift**: Monitor metrics on new data
- **Calibration Drift**: Track ECE over time
- **Data Quality**: Validate input feature distributions
- **Feedback Loops**: Incorporate clinical outcomes

### Retraining Triggers

Retrain model when:

- ROC-AUC drops below 0.85
- ECE exceeds 0.1 (poor calibration)
- Significant distribution shift detected
- New clinical evidence available

## References

1. Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning.
2. Guo, C. et al. (2017). On calibration of modern neural networks.
3. Clinical prediction models: a practical approach to development, validation, and updating (Steyerberg, 2019).