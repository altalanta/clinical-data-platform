from pathlib import Path

import duckdb

from clinical_platform.warehouse.loaders import init_warehouse


def test_init_warehouse_creates_tables(tmp_path: Path, monkeypatch):
    # Point to a temp DB
    from clinical_platform import config as cfg

    monkeypatch.setattr(cfg, "get_config", lambda: cfg.AppConfig.load("configs/config.local.yaml"))
    con = init_warehouse()
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "dim_subject" in tables

