# CP-Tox Mini Model Card

## Overview
- **Intended use:** Educational demo of an end-to-end cell painting + chem tox screening workflow. Not for clinical decisions.
- **Model family:** Logistic regression with z-score scaling on engineered CP + chemical descriptors.
- **Data provenance:** Deterministic synthetic plates (32×32 TIFF) paired with an eight-sample Tox21-like CSV. Inputs are hash-pinned via `manifests/data_manifest.json`.

## Metrics
- AUROC: {auroc}
- Average Precision: {ap}
- Brier score: {brier}
- Expected Calibration Error (10 bins): {ece}
- Train samples: {n_train}
- Test samples: {n_test}

## Calibration
Calibration is evaluated via reliability curves; see `reports/figures/calibration.png` for the full plot.

## Leakage & Batch Effects
- Risk level: **{leakage_risk}**
- Notes: {leakage_note}

## Known limitations
- Dataset is tiny and synthetic; metrics will not translate to production.
- Model uses simple descriptors and does not capture richer chemistry or imaging features.
- Leakage probes rely on linear classifiers and may miss non-linear effects.

## Update policy
- Regenerate reports by running `make all`; outputs are deterministic with `PYTHONHASHSEED=0` and `random_state=42`.
- Publish updated docs via CI to the `docs/` GitHub Pages site.
