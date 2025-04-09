import os
from unittest import mock
from manubot_ai_editor import env_vars
import pytest

from manubot_ai_editor.model_providers import MODEL_PROVIDERS


@pytest.mark.parametrize(
    "provider",
    MODEL_PROVIDERS.values(),
)
def test_model_provider_fields(provider):
    """
    Tests that each model provider has:
    - a default model engine
    - a non-empty API key environment variable (if applicable)
    - clients for the 'chat' and 'completions' endpoints
    - an endpoint for the default model engine
    - at least one model available
    """

    # check that each provider has a default model engine
    default_engine = provider.default_model_engine()
    assert default_engine is not None and len(default_engine) > 0

    # check that for providers that have API keys, they're
    # set to a non-empty string
    api_key = provider.api_key_env_var()
    assert api_key is None or api_key.strip() != ""

    # test that each provider provides both the 'chat' and 'completions'
    # endpoints
    clients = provider.clients()
    assert set(clients.keys()) == {"chat", "completions"}

    # test that the endpoint for the default model engine is valid
    endpoint = provider.endpoint_for_model(default_engine)
    assert endpoint in clients

    # check that there's at least one model available from the provider
    assert len(provider.get_models()) > 0


@pytest.mark.parametrize(
    "provider",
    MODEL_PROVIDERS.values(),
)
def test_model_provider_specific_key_resolution(provider):
    """
    Tests that the model provider correctly resolves a provider-specific API key
    from the environment variables. If it's not required by the provider, checks
    that the key is set to None.
    """

    api_key_var = provider.api_key_env_var()

    if api_key_var is None:
        # ensure that providers that don't require an API key
        # resolve None for the key
        assert provider.resolve_api_key() is None
    else:
        # if the provider does require a key, check that the provider-specific
        # key is resolved
        with mock.patch.dict("os.environ", {api_key_var: "1234"}):
            assert provider.resolve_api_key() == "1234"


@pytest.mark.parametrize(
    "provider",
    MODEL_PROVIDERS.values(),
)
@mock.patch.dict("os.environ", {env_vars.PROVIDER_API_KEY: "1234"})
def test_model_provider_generic_key_resolution(provider):
    """
    Tests that the model provider correctly resolves the generic API key from
    the environment variables in the absence of a provider-specific key. If it's
    not required by the provider, checks that the key is set to None.
    """

    with mock.patch.dict("os.environ"):
        # remove the provider-specific key to make sure we're checking generic
        # key resolution
        if (key := provider.api_key_env_var()) is not None:
            del os.environ[key]

        # check that the generic key is used
        if provider.api_key_env_var() is not None:
            assert provider.resolve_api_key() == "1234"
        else:
            assert provider.resolve_api_key() is None


@pytest.mark.parametrize(
    "provider",
    MODEL_PROVIDERS.values(),
)
@mock.patch.dict("os.environ", {env_vars.PROVIDER_API_KEY: "1234"})
def test_model_provider_get_models(provider):
    """
    Tests that the model provider can correctly retrieve the list of models
    from the cache, and that the default language model for each provider
    is in that list.
    """

    with mock.patch.dict("os.environ"):
        # remove the provider-specific key to ensure that a valid key doesn't
        # interfere with checking the cache
        if (key := provider.api_key_env_var()) is not None:
            del os.environ[key]

        # check that we can find the default model in each provider's list of
        # models
        default_model = provider.default_model_engine()
        assert default_model is not None
        assert default_model in provider.get_models()
