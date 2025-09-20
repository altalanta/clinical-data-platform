import pandas as pd

from clinical_platform.validation.pandera_models import AEModel, DMModel


def test_dm_schema_strict():
    df = pd.DataFrame({
        'STUDYID': ['S'], 'SUBJID': ['SUBJ1'], 'ARM': ['ACTIVE'], 'SEX': ['M'], 'AGE': [55]
    })
    validated = DMModel.validate(df)
    assert len(validated) == 1


def test_ae_schema_basic():
    df = pd.DataFrame({
        'STUDYID': ['S'], 'SUBJID': ['SUBJ1'], 'AESTDTC': ['2024-01-01'], 'AEENDTC': ['2024-01-02'],
        'AESEV': ['MILD'], 'AESER': [True], 'AEOUT': ['RECOVERED']
    })
    validated = AEModel.validate(df)
    assert len(validated) == 1

