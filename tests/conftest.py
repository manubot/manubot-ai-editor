"""
Configures 'cost' marker for tests that cost money (i.e. OpenAI API credits)
to run.

Adapted from https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
"""

import json
import pytest

from pathlib import Path
from unittest import mock


def pytest_addoption(parser):
    parser.addoption(
        "--runcost",
        action="store_true",
        default=False,
        help="run tests that can incur API usage costs",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cost: mark test as possibly costing money to run"
    )
    config.addinivalue_line(
        "markers", "mocked_model_list: mark test as having used the provider's cached model list"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runcost"):
        # --runcost given in cli: do not skip cost tests
        return

    skip_cost = pytest.mark.skip(reason="need --runcost option to run")

    for item in items:
        if "cost" in item.keywords:
            item.add_marker(skip_cost)


# since we don't have valid provider API keys during tests, we can't get the
# model list from the provider API. instead, we mock the method that retrieves
# the model list to return a cached version of the model list stored in the
# provider_model_engines.json file
@pytest.fixture(autouse=True, scope="function")
def patch_model_list_cache(request):
    # skip patching if the test or anything above it is marked with 'cost',
    # which implies we have a valid API key and thus should retrieve the model
    # list from the provider API
    if request.node.get_closest_marker("cost") is not None:
        yield
        return

    # path to the provider_model_engine.json file
    provider_model_engine_json = (
        Path(__file__).parent / "provider_fixtures" / "provider_model_engines.json"
    )

    # load the provider model list once, then use it in our mocked method
    with provider_model_engine_json.open("r") as f:
        provider_model_engines = json.load(f)

    @classmethod
    def cached_model_list_retriever(cls):
        # annotate the request object to indicate we're using a mocked method
        # we want the live API tests to ensure they're not using this mock
        request.node.add_marker("mocked_model_list")

        return provider_model_engines[cls.__name__]

    # finally, apply the mock
    with mock.patch(
        "manubot_ai_editor.model_providers.BaseModelProvider.get_models",
        new=cached_model_list_retriever,
    ) as mock_method:
        yield mock_method
