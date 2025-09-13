# ML Slice

- Problem: predict early dropout or serious AE risk from subject-level features
- Pipeline: feature aggregation → split/train (CV) → evaluate AUROC/PR-AUC → log to MLflow
- Reproducibility: fixed seeds, logged params/artifacts, model signature
- Monitoring: data drift (summary stats), performance drift (holdout metrics), simple alerts

