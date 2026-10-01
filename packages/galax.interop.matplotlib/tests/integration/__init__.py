"""Test cases for the matplotlib integration.

This uses the `pytest-mpl` plugin to compare the generated plots with the
expected ones.

To generate the expected plots, run the following command:

    pytest packages/galax.interop.matplotlib/tests/integration \
        --mpl-generate-path=packages/galax.interop.matplotlib/tests/integration/baseline
    pytest packages/galax.interop.matplotlib/tests/integration \
        --mpl-generate-hash-library=packages/galax.interop.matplotlib/tests/integration/hashes.json

"""
