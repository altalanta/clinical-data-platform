# Model Card

## Clinical Risk Prediction Model v1.0.0

This repository contains a clinical risk prediction model designed for adverse event risk stratification in clinical trial settings.

### Quick Overview

- **Algorithm**: Logistic Regression with L2 regularization
- **Task**: Binary classification (adverse event prediction)
- **Performance**: ROC-AUC 0.918 ± 0.012 (5-fold CV)
- **Calibration**: Well-calibrated (ECE = 0.034)
- **Data**: Synthetic OMOP CDM v6.0 (1,000 patients)

### Key Features

- **Primary Features**: AGE, AE_COUNT, SEVERE_AE_COUNT
- **HIPAA Compliance**: Full PHI redaction and secure API
- **Production Ready**: Comprehensive evaluation with calibration analysis
- **Quality Gates**: 80% test coverage, type checking, linting

### Documentation

📋 **[Complete Model Card Documentation](docs/model_card.md)**

The full model card includes:
- Detailed performance metrics with confidence intervals
- Calibration analysis and reliability diagrams
- Clinical decision thresholds and use cases
- Limitations, bias considerations, and ethical guidelines
- PHI handling and HIPAA compliance features
- Deployment guidelines and monitoring recommendations

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run model evaluation
python -m clinical_data_platform.eval --output-dir artifacts/eval

# Start API server
uvicorn clinical_platform.api.main:app --host 0.0.0.0 --port 8000

# Generate predictions
curl -X POST http://localhost:8000/score \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"AGE": 65.0, "AE_COUNT": 3.0, "SEVERE_AE_COUNT": 1.0}'
```

### Repository Structure

```
docs/
├── model_card.md          # Complete model documentation
├── model_evaluation.md    # Evaluation methodology and results
├── api.md                 # API reference and examples
└── data_warehouse.md      # Data pipeline and dbt documentation

src/clinical_data_platform/
├── eval.py               # Model evaluation pipeline
└── models/               # ML model implementations

artifacts/eval/
├── cv_metrics.json       # Cross-validation results
└── calibration.png       # Calibration plot
```

### Citation

```bibtex
@software{clinical_risk_prediction_model,
  title={Clinical Risk Prediction Model},
  author={Clinical Data Platform Team},
  year={2024},
  version={1.0.0},
  url={https://github.com/altalanta/clinical-data-platform}
}
```

---

**⚠️ Important**: This model is trained on synthetic data and is intended for demonstration purposes. Clinical validation on real patient populations is required before deployment in clinical settings.

For technical support: clinical-platform-support@organization.com