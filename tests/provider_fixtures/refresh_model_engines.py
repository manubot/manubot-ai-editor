#!/usr/bin/env python

"""
This command persists model engines for each provider to MODEL_PROVIDER_JSON.
This list is used in non-live tests so that we don't need users who run the test
suite to provide valid API keys for every provider.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from manubot_ai_editor.model_providers import MODEL_PROVIDERS

MODEL_PROVIDER_JSON = "provider_model_engines.json"


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

    with Path(MODEL_PROVIDER_JSON).open("w") as f:
        model_list = {
            provider.__class__.__name__: provider.get_models()
            for provider in MODEL_PROVIDERS.values()
            if not provider.is_local_provider()
        }
        model_list["__generated_on__"] = datetime.now().isoformat()

        json.dump(model_list, f, indent=2)

        return model_list


def retrieve_provider_model_engines():
    """
    Pulls the default model engines for each provider from the JSON file
    distributed with the package.
    """

    with Path(MODEL_PROVIDER_JSON).open("r") as f:
        return json.load(f)


def main():
    # check if we have valid API keys for each provider
    for provider in [x for x in MODEL_PROVIDERS.values() if x.is_local_provider()]:
        provider_key_var = provider.api_key_env_var()

        if not os.environ.get(provider_key_var):
            raise ValueError(
                f"Provider {provider.__class__.__name__} requires an API key in"
                f" env var {provider_key_var}, but none is set."
            )

    # persist the provider model engines to a JSON file
    new_list = persist_provider_model_engines()

    print(
        f"Persisted {sum(len(x) for x in new_list.values())} model engines"
        f" to {MODEL_PROVIDER_JSON}"
    )


if __name__ == "__main__":
    main()
