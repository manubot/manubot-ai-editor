"""
This module defines metadata for the model providers that we're accessing via
LangChain.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
import os

from langchain_anthropic import Anthropic, ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAI

from manubot_ai_editor import env_vars

from logging import getLogger

logger = getLogger(__name__)


# =============================================================================
# === Provider exceptions
# =============================================================================


class APIKeyNotFoundError(Exception):
    """
    Raised when an API key is required by a provider but not found in the
    environment variables.
    """

    pass


class APIModelListNotObtainable(Exception):
    """
    Raised when a model list cannot be obtained from the provider.

    This is raised in response to APIError, which will most often
    occur from an invalid API key.
    """

    def __init__(self, provider):
        self.provider = provider
        super().__init__()


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
    def get_models(cls):
        """
        Returns a list of models available from this provider.

        Override this method to return None if your provider doesn't have a
        way of providing a list of models.

        Raises:
            APIModelListNotObtainable:
                If the model list cannot be obtained from the provider. This is
                most often due to an invalid API key, but could also be due to
                the API being unavailable for some other reason.
            ImplicitDependencyImportError:
                If the provider-specific library cannot be imported. This is
                most often due to the library not being installed, but could
                also be due to the library's internal structure changing such
                that we can't find its model-fetching function.
        """
        return cls._get_provider_models()

    @classmethod
    @abstractmethod
    def _get_provider_models(cls):
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
    @lru_cache(maxsize=None)
    def _get_provider_models(cls):
        try:
            import openai

            client = openai.OpenAI(api_key=cls.resolve_api_key())

            return [model.id for model in client.models.list().data]

        except openai.APIError as ex:
            raise APIModelListNotObtainable(provider=cls.__name__) from ex

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
    @lru_cache(maxsize=None)
    def _get_provider_models(cls):
        try:
            import anthropic

            client = anthropic.Client(api_key=cls.resolve_api_key())

            return [model.id for model in client.models.list().data]

        except (anthropic.APIError, TypeError) as ex:
            raise APIModelListNotObtainable(provider=cls.__name__) from ex

        except ImportError:
            raise ImplicitDependencyImportError(
                langchain_module="langchain_anthropic", implicit_dependency="anthropic"
            )


# the MODEL_PROVIDERS dict specifies metadata for each model provider, e.g.
# OpenAI or Anthropic, that are used in the GPT3CompletionModel class to get
# API keys, client instances, and other metadata for each provider
MODEL_PROVIDERS = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
}
