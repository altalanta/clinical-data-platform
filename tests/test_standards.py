import pandas as pd

from clinical_platform.standards.cdisc_sdtm_mapping import map_dm, map_ae


def test_map_dm_basic():
    df = pd.DataFrame({
        'STUDYID': ['S'], 'SUBJID': ['SUBJ0001'], 'ARM': ['ACTIVE'], 'SEX': ['M'], 'AGE': [50]
    })
    out = map_dm(df)
    assert set(out.columns) == {'STUDYID', 'SUBJID', 'ARM', 'SEX', 'AGE'}


def test_map_ae_types():
    df = pd.DataFrame({
        'STUDYID': ['S'], 'SUBJID': ['SUBJ0001'], 'AESTDTC': ['2024-01-01'], 'AEENDTC': ['2024-01-02'],
        'AESEV': ['MILD'], 'AESER': [True], 'AEOUT': ['RECOVERED']
    })
    out = map_ae(df)
    assert str(out['AESTDTC'].dtype).startswith('datetime64')
    assert str(out['AEENDTC'].dtype).startswith('datetime64')
    assert out['AESER'].dtype.name == 'boolean'

