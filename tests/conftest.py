"""
Configures 'cost' marker for tests that cost money (i.e. OpenAI API credits)
to run.

Adapted from https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runcost", action="store_true", default=False, help="run tests that can incur API usage costs"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cost: mark test as possibly costing money to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runcost"):
        # --runcost given in cli: do not skip cost tests
        return
    
    skip_cost = pytest.mark.skip(reason="need --runcost option to run")

    for item in items:
        if "cost" in item.keywords:
            item.add_marker(skip_cost)
