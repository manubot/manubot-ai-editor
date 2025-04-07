"""
This module defines metadata for the model providers that we're accessing via
LangChain.
"""

import json

from datetime import datetime
from abc import ABC
from functools import lru_cache
import os

from langchain_anthropic import Anthropic, ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAI

from manubot_ai_editor import env_vars


# =============================================================================
# === Provider exceptions
# =============================================================================


class APIKeyNotFoundError(Exception):
    """
    Raised when an API key is required by a provider but not found in the
    environment variables.
    """

    pass


class ImplicitDependencyImportError(ImportError):
    """
    Raised when we attempt to import an implicit dependency for a provider but
    fail to do so.

    Since LangChain doesn't abstract fetching model engine lists for providers,
    we instead have to rely on the provider-specific package to do that. These
    provider-specific packages are currently implicit dependencies of the
    langchain community modules for each provider, but this may change, thus why
    we go to such an effort to report it.
    """

    def __init__(self, langchain_module, implicit_dependency):
        self.message = (
            f"Failed to import '{implicit_dependency}' via implicit dependency in "
            f"'{langchain_module}'. The implicit dependency is required to fetch "
            f"a list of models."
        )
        super().__init__(self.message)


# =============================================================================
# === Base and specific provider classes
# =============================================================================

# decorator to cache, e.g., model lists we pull from the APIs
cache = lru_cache(maxsize=None)


class BaseModelProvider(ABC):
    """
    Base class for model providers. Provides methods to acquire an API key, if
    necessary, a client instance for the 'chat' and 'completions' endpoints, and
    a list of models available from the provider.
    """

    @classmethod
    def resolve_api_key(cls, given_api_key=None):
        """
        Returns an API key for this provider via the given key, then the
        provider-specific env var for the key, then the generic key env var. If
        none of that works, it raises an APIKeyNotFoundError.

        If the provider doesn't require an API key (indicated by
        cls.api_key_env_var() being None), this returns None, indicating no key
        is required.

        If we've explicitly been given an API key, we'll use that. Otherwise,
        we'll check the environment variables for the provider-specific key,
        then the generic key, and raise an error if neither is set.
        """

        if given_api_key is not None:
            return given_api_key

        if (api_key_env_var := cls.api_key_env_var()) is not None:
            candidate_key = None

            # first check if the provider-specific key is set, and if not
            # check the generic key
            if (
                provider_specific_key := os.environ.get(api_key_env_var)
            ) is not None and provider_specific_key.strip() != "":
                candidate_key = provider_specific_key
            elif (
                generic_key := os.environ.get(env_vars.PROVIDER_API_KEY)
            ) is not None and generic_key.strip() != "":
                candidate_key = generic_key

            # if the key is empty, raise an error
            if candidate_key is None or candidate_key.strip() == "":
                raise APIKeyNotFoundError(
                    f"API key for provider {cls.__name__} is empty"
                )

            return candidate_key

        else:
            return None

    @classmethod
    def is_local_provider(cls):
        """
        Returns True if this provider is a local provider.

        Local providers don't need to persist their model
        list to the provider model engines JSON file and they
        *typically* don't require a key, but you should check
        that the api_key_env_var method returns None to be sure.
        """
        return False

    @classmethod
    @abstractmethod
    def default_model_engine(cls):
        """ "
        Returns a string that indicates the default
        model for this provider.
        """
        return NotImplementedError

    @classmethod
    def api_key_env_var(cls):
        """
        Returns the environment variable that should be used to
        obtain the API key for this provider. If this provider
        doesn't require an API key, this should return None.
        """
        return None

    @classmethod
    @abstractmethod
    def clients(cls):
        """ "
        Returns a dictionary of the form {'chat': ChatClient, 'completions':
        CompletionsClient} that maps the endpoints to the client classes that
        should be used to interact with the provider's API.
        """
        return NotImplementedError

    @classmethod
    @abstractmethod
    def endpoint_for_model(cls, model_engine):
        """
        Returns a client (e.g., 'chat' or 'completions') based on the endpoint
        for the model engine specified.
        """
        return NotImplementedError

    @classmethod
    def get_models(cls, use_local_cache=True):
        """
        Returns a list of models available from this provider. If for some
        reason the models couldn't be retrieved, this should return None.
        """
        return NotImplementedError


class OpenAIProvider(BaseModelProvider):
    @classmethod
    def default_model_engine(cls):
        return "gpt-4-turbo"

    @classmethod
    def api_key_env_var(cls):
        return env_vars.OPENAI_API_KEY

    @classmethod
    def clients(cls):
        return {"chat": ChatOpenAI, "completions": OpenAI}

    @classmethod
    def endpoint_for_model(cls, model_engine):
        if model_engine.startswith(
            ("text-davinci-", "text-curie-", "text-babbage-", "text-ada-")
        ):
            return "completions"
        else:
            return "chat"

    @classmethod
    @cache
    def get_models(cls, use_local_cache=True):
        try:
            import openai

            try:
                print(
                    f"Retrieving models from the OpenAI API using key {cls.resolve_api_key()}"
                )
                client = openai.OpenAI(api_key=cls.resolve_api_key())
                models = client.models.list()

                return [model.id for model in models.data]

            except openai.APIError as ex:
                # pull from local
                if use_local_cache:
                    print(
                        f"Unable to retrieve models from the API: {ex}, resorting to local cache"
                    )
                    return retrieve_provider_model_engines()["OpenAIProvider"]
                else:
                    raise ex

        except ImportError:
            raise ImplicitDependencyImportError(
                langchain_module="langchain_openai", implicit_dependency="openai"
            )


class AnthropicProvider(BaseModelProvider):
    @classmethod
    def default_model_engine(cls):
        return "claude-3-haiku-20240307"

    @classmethod
    def api_key_env_var(cls):
        return env_vars.ANTHROPIC_API_KEY

    @classmethod
    def clients(cls):
        return {"chat": ChatAnthropic, "completions": Anthropic}

    @classmethod
    def endpoint_for_model(cls, model_engine):
        if model_engine.startswith(("claude-2",)):
            return "completions"
        else:
            return "chat"

    @classmethod
    @cache
    def get_models(cls, use_local_cache=True):
        try:
            import anthropic

            try:
                client = anthropic.Client(api_key=cls.resolve_api_key())
                models = client.models.list()

                return [model.id for model in models.data]

            except (anthropic.APIError, TypeError) as ex:
                # pull from local
                if use_local_cache:
                    print(
                        f"Unable to retrieve models from the API: {ex}, resorting to local cache"
                    )
                    return retrieve_provider_model_engines()["AnthropicProvider"]
                else:
                    raise ex

        except ImportError:
            raise ImplicitDependencyImportError(
                langchain_module="langchain_anthropic", implicit_dependency="anthropic"
            )


# =============================================================================
# === Provider dict, model engine fallback list persistence + retrieval
# =============================================================================

# the MODEL_PROVIDERS dict specifies metadata for each model provider, e.g.
# OpenAI or Anthropic, that are used in the GPT3CompletionModel class to invoke
# the provider's API.
# - the 'api_key_env_var' field specifies the environment variable that should be
#   used to obtain the API key for the provider. if it's None, that means this
#   provider doesn't require an API key (e.g., for local LLMs)
# - the 'clients' field maps the endpoints to the client classes that should be
#   used to interact with the provider's API.
MODEL_PROVIDERS = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
}


def _provider_model_engine_file():
    """
    Returns the path to the JSON file that contains the available model engines
    for each provider.
    """
    from importlib import resources

    # form path to the JSON file
    model_path = resources.files("manubot_ai_editor").joinpath(
        "ref", "provider_model_engines.json"
    )

    # ensure the directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    return model_path


def persist_provider_model_engines():
    """
    Persists the default model engines for each provider to a JSON file,
    distributed with the package. This method requires valid API keys to be
    available for each provider, since the list of models per provider is pulled
    from the API.

    The JSON file is used as as a fallback in case the API is for some reason
    unavailable, e.g. in testing when we don't have valid keys.

    It's unfortunate that the model providers require authenticated access to
    get the list of models, but that's how it is. Ideally we'd run this with
    each release, to at least capture which model engines are available at the
    time of release.
    """

    with _provider_model_engine_file().open("w") as f:
        model_list = {
            provider.__class__.__name__: provider.get_models()
            for provider in MODEL_PROVIDERS.values()
            if not provider.is_local_provider()
        }
        model_list["__generated_on__"] = datetime.now().isoformat()

        json.dump(model_list, f, indent=2)


def retrieve_provider_model_engines():
    """
    Pulls the default model engines for each provider from the JSON file
    distributed with the package.
    """

    with _provider_model_engine_file().open("r") as f:
        return json.load(f)
