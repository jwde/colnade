"""Smoke tests — verify packages are importable."""


def test_import_colnade() -> None:
    import colnade

    assert colnade is not None


def test_import_colnade_polars() -> None:
    import colnade_polars

    assert colnade_polars is not None


def test_import_colnade_pandas() -> None:
    import colnade_pandas

    assert colnade_pandas is not None


def test_import_colnade_dask() -> None:
    import colnade_dask

    assert colnade_dask is not None
