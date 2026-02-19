"""Nox sessions for testing against multiple backend versions."""

import nox

nox.options.default_venv_backend = "uv"

POLARS_VERSIONS = ["1.0.0", "1.20.0"]
PANDAS_VERSIONS = ["2.0.0", "2.2.0"]
DASK_VERSIONS = ["2024.1.0", "2024.6.0"]

POLARS_TESTS = [
    "tests/integration/test_polars_execution.py",
    "tests/integration/test_polars_io.py",
    "tests/integration/test_validation_polars.py",
    "tests/integration/test_field_validation_polars.py",
]

PANDAS_TESTS = [
    "tests/integration/test_pandas_execution.py",
    "tests/integration/test_pandas_io.py",
    "tests/integration/test_field_validation_pandas.py",
]

DASK_TESTS = [
    "tests/integration/test_dask_execution.py",
    "tests/integration/test_dask_io.py",
    "tests/integration/test_field_validation_dask.py",
]


@nox.session(python=["3.10"])
@nox.parametrize("polars", POLARS_VERSIONS)
def test_polars(session: nox.Session, polars: str) -> None:
    """Test colnade-polars against specific Polars versions."""
    session.install("-e", ".", "-e", "colnade-polars", "pytest", f"polars=={polars}")
    session.run("pytest", *POLARS_TESTS, "-q")


@nox.session(python=["3.10"])
@nox.parametrize("pandas", PANDAS_VERSIONS)
def test_pandas(session: nox.Session, pandas: str) -> None:
    """Test colnade-pandas against specific Pandas versions."""
    deps = ["-e", ".", "-e", "colnade-pandas", "pytest", f"pandas=={pandas}"]
    # pandas < 2.2 was compiled against numpy 1.x ABI
    if pandas < "2.2":
        deps.append("numpy<2")
    session.install(*deps)
    session.run("pytest", *PANDAS_TESTS, "-q")


@nox.session(python=["3.10"])
@nox.parametrize("dask", DASK_VERSIONS)
def test_dask(session: nox.Session, dask: str) -> None:
    """Test colnade-dask against specific Dask versions."""
    session.install(
        "-e",
        ".",
        "-e",
        "colnade-pandas",
        "-e",
        "colnade-dask",
        "pytest",
        f"dask[dataframe]=={dask}",
    )
    session.run("pytest", *DASK_TESTS, "-q")
