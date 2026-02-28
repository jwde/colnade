# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/jwde/colnade/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | ------: | --------: |
| colnade-dask/src/colnade\_dask/\_\_init\_\_.py     |        5 |        0 |    100% |           |
| colnade-dask/src/colnade\_dask/adapter.py          |      410 |       16 |     96% |132-134, 174, 369-370, 389-390, 399-400, 405, 424-425, 439, 445, 522 |
| colnade-dask/src/colnade\_dask/io.py               |       44 |        0 |    100% |           |
| colnade-pandas/src/colnade\_pandas/\_\_init\_\_.py |        6 |        0 |    100% |           |
| colnade-pandas/src/colnade\_pandas/adapter.py      |      402 |       11 |     97% |358-359, 378-379, 391-392, 398, 416-417, 455, 510 |
| colnade-pandas/src/colnade\_pandas/conversion.py   |       28 |        0 |    100% |           |
| colnade-pandas/src/colnade\_pandas/io.py           |       39 |        0 |    100% |           |
| colnade-polars/src/colnade\_polars/\_\_init\_\_.py |        6 |        0 |    100% |           |
| colnade-polars/src/colnade\_polars/adapter.py      |      320 |        1 |     99% |       359 |
| colnade-polars/src/colnade\_polars/conversion.py   |       37 |        0 |    100% |           |
| colnade-polars/src/colnade\_polars/io.py           |       48 |        0 |    100% |           |
| src/colnade/\_\_init\_\_.py                        |       11 |        0 |    100% |           |
| src/colnade/\_protocols.py                         |        6 |        0 |    100% |           |
| src/colnade/\_types.py                             |        7 |        0 |    100% |           |
| src/colnade/arrow.py                               |       30 |        0 |    100% |           |
| src/colnade/constraints.py                         |       54 |        0 |    100% |           |
| src/colnade/dataframe.py                           |      486 |        3 |     99% |   638-640 |
| src/colnade/dtypes.py                              |       27 |        0 |    100% |           |
| src/colnade/expr.py                                |      214 |        0 |    100% |           |
| src/colnade/schema.py                              |      326 |        1 |     99% |       192 |
| src/colnade/validation.py                          |       93 |        1 |     99% |       196 |
| **TOTAL**                                          | **2599** |   **33** | **99%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/jwde/colnade/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/jwde/colnade/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/jwde/colnade/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/jwde/colnade/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fjwde%2Fcolnade%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/jwde/colnade/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.