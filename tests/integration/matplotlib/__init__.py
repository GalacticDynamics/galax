"""Test cases for the matplotlib integration.

This uses the `pytest-mpl` plugin to compare the generated plots with the
expected ones.

To generate the expected plots, run the following command:

    pytest tests/integration --mpl-generate-path=tests/integration/matplotlib/baseline
    pytest tests/integration --mpl-generate-hash-library=tests/integration/matplotlib/hashes.json

"""  # noqa: E501
