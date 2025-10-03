from clinical_platform.ml.train import train


def test_train_returns_metrics():
    auc, ap = train(data_dir=None, out_dir="models-test")
    assert 0 <= auc <= 1
    assert 0 <= ap <= 1

