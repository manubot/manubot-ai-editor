import os
import re
from abc import ABC, abstractmethod
import random
import time
import json

from logging import getLogger

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from manubot_ai_editor import env_vars
from manubot_ai_editor.exceptions import APIKeyInvalidError
from manubot_ai_editor.model_providers import (
    MODEL_PROVIDERS,
    APIKeyNotFoundError,
    APIModelListNotObtainable,
)


logger = getLogger(__name__)


class ManuscriptRevisionModel(ABC):
    """
    An abstract class for manuscript revision models.
    """

    def __init__(self):
        pass

    @abstractmethod
    def revise_paragraph(self, paragraph_text, section_name, resolved_prompt=None):
        """
        It revises a paragraph of a manuscript from a given section.

        Args:
            paragraph_text (str): text of the paragraph to revise.
            section_name (str): name of the section the paragraph belongs to.
            resolved_prompt (str): prompt resolved via ai-revision config files, if available

        Returns:
            Revised paragraph text.
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, paragraph_text, section_name, resolved_prompt: str = None):
        """
        Returns the prompt to be used for the revision of a paragraph that
        belongs to a given section.
        """
        raise NotImplementedError


class DummyManuscriptRevisionModel(ManuscriptRevisionModel):
    """
    This model does nothing, just returns the same paragraph content with
    sentences one after the other separated by a white space (no new lines).
    """

    def __init__(self, add_paragraph_marks=False):
        super().__init__()
        self.sentence_end_pattern = re.compile(r".\n")
        self.add_paragraph_marks = add_paragraph_marks

    def revise_paragraph(self, paragraph_text, section_name, resolved_prompt=None):
        if self.add_paragraph_marks:
            return (
                "%%% PARAGRAPH START %%%\n"
                + paragraph_text.strip()
                + "\n%%% PARAGRAPH END %%%"
            )

        return self.sentence_end_pattern.sub(". ", paragraph_text).strip()

    def get_prompt(self, paragraph_text, section_name, resolved_prompt: str = None):
        return paragraph_text


class VerboseManuscriptRevisionModel(DummyManuscriptRevisionModel):
    """
    This model returns the same paragraph and adds a header to it.
    """

    def __init__(self, revised_header: str = "Revised:"):
        super().__init__()
        self.revised_header = revised_header

    def revise_paragraph(self, paragraph_text, section_name, resolved_prompt=None):
        revised_paragraph = super().revise_paragraph(paragraph_text, section_name)
        return f"{self.revised_header}{revised_paragraph}"


class RandomManuscriptRevisionModel(ManuscriptRevisionModel):
    """
    This model takes a paragraph and randomizes the words. The paragraph has the
    sentences one after the other separated by a white space (no new lines).
    """

    def __init__(self):
        super().__init__()
        self.sentence_end_pattern = re.compile(r"\n")

    def revise_paragraph(
        self, paragraph_text: str, section_name: str, resolved_prompt=None
    ) -> str:
        """
        It takes each sentence of the paragraph and randomizes the words.
        """
        paragraph_text = self.sentence_end_pattern.sub(" ", paragraph_text).strip()
        sentences = paragraph_text.split(". ")
        sentences_revised = []
        for sentence in sentences:
            words = sentence.split(" ")
            words_revised = []
            for word in words:
                if len(word) > 3:
                    word_revised = (
                        "".join(random.sample(word[1:-1], len(word[1:-1]))) + word[-1]
                    )
                else:
                    word_revised = word
                words_revised.append(word_revised)
            sentences_revised.append(" ".join(words_revised))
        return ". ".join(sentences_revised)

    def get_prompt(self, paragraph_text, section_name, resolved_prompt: str = None):
        return paragraph_text


class GPT3CompletionModel(ManuscriptRevisionModel):
    """
    Revises paragraphs using completion or chat completion models. Most of the parameters
    (https://platform.openai.com/docs/guides/gpt) of the model can be specified either by
    arguments during instantiation or environment variables (see env_vars.py).

    Note that a few arguments override the environment variables, such as
    'model_provider', 'model engine', and 'api_key'; the environment is only
    consulted when they're set to None. The rest of the parameters, e.g.
    'temperature', are overridden *by* the corresponding environment variables,
    if they exist.

    Regarding temperature, best_of, top_p, etc., this post provides a good
    explanation of how they're related:
    https://community.openai.com/t/the-relationship-between-best-of-temperature-and-top-p-the-three-variable-problem/21150/11

    Args:
        title (str): Title of the manuscript.
        keywords (list): Keywords of the manuscript, defaults to [] if unspecified.
        model_provider (str):
            Model provider to use, e.g. "openai" or "anthropic". If not
            specified, it will be obtained from the environment variable
            specified by env_vars.MODEL_PROVIDER, defaulting to "openai" if not set.
        model_engine (str):
            Language model to use. For example, "text-davinci-003",
            "gpt-3.5-turbo", "gpt-3.5-turbo-0301", etc. If not specified, it
            will be obtained from the environment variable specified by
            env_vars.LANGUAGE_MODEL; failing that the default model for the
            provider will be used.
        api_key (str):
            API key for the model provider. If not specified, it will be
            obtained from the environment variable specified by the provider's
            API key env var (e.g. env_vars.OPENAI_API_KEY for OpenAI); failing
            that, it will be obtained from the generic PROVIDER_API_KEY env var.
        temperature (float):
            Temperature parameter for the model. If env_vars.TEMPERATURE is
            set, it will override this value.
        presence_penalty (float):
            Presence penalty parameter for the model. If env_vars.PRESENCE_PENALTY
            is set, it will override this value.
        frequency_penalty (float):
            Frequency penalty parameter for the model. If env_vars.FREQUENCY_PENALTY
            is set, it will override this value.
        best_of (int):
            Number of completions to generate and rank. If env_vars.BEST_OF
            is set, it will override this value.
        top_p (float):
            Top-P parameter for the model. If env_vars.TOP_P is set, it will
            override this value.
        retry_count (int):
            Number of times to retry the revision if an error occurs. If
            env_vars.RETRY_COUNT is set, it will override this value.
    """

    def __init__(
        self,
        title: str,
        keywords: list[str],
        model_provider: str = None,
        api_key: str = None,
        model_engine: str = None,
        temperature: float = 0.5,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        best_of: int = None,
        top_p: float = None,
        retry_count: int = 5,
    ):
        super().__init__()

        # if no model_provider was provided, get it from the environment,
        # defaulting to openai if not set
        if model_provider is None:
            model_provider = os.environ.get(env_vars.MODEL_PROVIDER, "openai")

        # now, get the metadata object for the model provider
        try:
            provider = MODEL_PROVIDERS[model_provider]
        except KeyError:
            raise ValueError(
                f"Model provider '{model_provider}' not found; it must be one of {', '.join(MODEL_PROVIDERS.keys())}"
            )

        # identify model_engine first by the argument, then by the environment
        # var env_vars.LANGUAGE_MODEL, then by whatever the provider's default
        # model is
        if model_engine is None or model_engine.strip() == "":
            model_engine = os.environ.get(env_vars.LANGUAGE_MODEL)

            # if it's *still* None or empty, use the provider's default
            if model_engine is None or model_engine.strip() == "":
                model_engine = provider.default_model_engine()

                # report that we resorted to the provider's default model engine
                print(
                    f"Using default language model '{model_engine}' for provider '{model_provider}'"
                )
            else:
                # report that we were able to retrieve a model engine from the environment
                print(
                    f"Using language model '{model_engine}' from environment variable '{env_vars.LANGUAGE_MODEL}'"
                )

        # if no api_key was explicitly provided and the provider requires an API
        # key, make sure a key can be found
        if (
            api_key is None
            and (provider_key_env_var := provider.api_key_env_var()) is not None
        ):
            # consult the provider for a possible key
            try:
                api_key = provider.resolve_api_key()
            except APIKeyNotFoundError:
                raise ValueError(
                    f"API key for provider {model_provider} not found. Please provide it as the 'api_key' parameter, "
                    f"set a provider-specific key via the environment variable {provider_key_env_var}, "
                    f"or set a generic API key via the environment variable {env_vars.PROVIDER_API_KEY}",
                )

        if env_vars.TEMPERATURE in os.environ:
            try:
                temperature = float(os.environ[env_vars.TEMPERATURE])
                print(
                    f"Using temperature from environment variable '{env_vars.TEMPERATURE}'"
                )
            except ValueError:
                # if it is not a float, we ignore it
                pass

        if env_vars.TOP_P in os.environ:
            try:
                top_p = float(os.environ[env_vars.TOP_P])
                print(f"Using top_p from environment variable '{env_vars.TOP_P}'")
            except ValueError:
                # if it is not a float, we ignore it
                pass

        if env_vars.PRESENCE_PENALTY in os.environ:
            try:
                presence_penalty = float(os.environ[env_vars.PRESENCE_PENALTY])
                print(
                    f"Using presence_penalty from environment variable '{env_vars.PRESENCE_PENALTY}'"
                )
            except ValueError:
                # if it is not a float, we ignore it
                pass

        if env_vars.FREQUENCY_PENALTY in os.environ:
            try:
                frequency_penalty = float(os.environ[env_vars.FREQUENCY_PENALTY])
                print(
                    f"Using frequency_penalty from environment variable '{env_vars.FREQUENCY_PENALTY}'"
                )
            except ValueError:
                # if it is not a float, we ignore it
                pass

        if env_vars.BEST_OF in os.environ:
            try:
                best_of = int(os.environ[env_vars.BEST_OF])
                print(f"Using best_of from environment variable '{env_vars.BEST_OF}'")
            except ValueError:
                # if it is not an int, we ignore it
                pass

        self.retry_count = retry_count
        if env_vars.RETRY_COUNT in os.environ:
            try:
                self.retry_count = int(os.environ[env_vars.RETRY_COUNT])
                print(
                    f"Using retry_count from environment variable '{env_vars.RETRY_COUNT}'"
                )
            except ValueError:
                # if it is not an int, we ignore it
                pass

        self.title = title
        self.keywords = keywords if keywords is not None else []

        # set the endpoint ('chat' or 'completions') based on the provider's
        # specific definition of which models support what endpoints
        # (this is almost alway 'chat', but some legacy models only
        # work with 'completions')
        self.endpoint = provider.endpoint_for_model(model_engine)

        # emit information about the model provider, engine, and endpoint
        # (this is scraped from the output by the rootstock ai-revision
        # workflow, so keep the prefix text exactly as-is or update it in that
        # workflow.)
        print(f"Model provider: {model_provider}")
        print(f"Language model: {model_engine}")
        print(f"Model endpoint used: {self.endpoint}")

        self.model_parameters = {
            "model": model_engine,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "best_of": best_of,
        }

        # keep model parameters that are not None only
        self.model_parameters = {
            key: value
            for key, value in self.model_parameters.items()
            if value is not None
        }

        self.several_spaces_pattern = re.compile(r"\s+")

        # ensure that the model engine we selected is available for the
        # provider we selected
        try:
            models = provider.get_models()

            if models is not None and model_engine not in models:
                raise ValueError(
                    f"Model engine '{model_engine}' is not available for provider '{model_provider}'; "
                    f"available models are: {', '.join(models)}"
                )
            elif models is None:
                logger.warning(
                    f"Provider '{model_provider}' declares it can't list models; "
                    f"assuming model '{model_engine}' is valid and continuing"
                )

        except APIModelListNotObtainable:
            logger.warning(
                f"Unable to obtain model list from provider '{model_provider}', assuming it's valid and continuing"
            )

        # construct the provider's client after all the rest of
        # the settings above have been processed
        client_cls = provider.clients()[self.endpoint]
        self._model_provider = model_provider
        self.client = client_cls(
            api_key=api_key,
            **self.model_parameters,
        )

    def get_prompt(
        self, paragraph_text: str, section_name: str = None, resolved_prompt: str = None
    ) -> str | tuple[str, str]:
        """
        Returns the prompt to be used for the revision of a paragraph that
        belongs to a given section. There are three types of prompts according
        to the section: Abstract, Introduction, Methods, and the rest (i.e.
        Results, Discussion, etc.).

        Args:
            paragraph_text: text of the paragraph to revise.
            section_name: name of the section the paragraph belongs to.
            resolved_prompt: prompt resolved via ai-revision config, if available

        Returns:
            A string with the prompt to be used by the model for the revision of the paragraph.
            It contains two paragraphs of text: the command for the model
            ("Revise...") and the paragraph to revise.
        """

        # prompts are resolved in the following order, with the first satisfied
        # condition taking effect:

        # 1. if a custom prompt is specified via the env var specified by
        #    env_vars.CUSTOM_PROMPT, then the text in that env var is used as
        #    the prompt.
        # 2. if the files ai-revision-config.yaml and/or ai-revision-prompt.yaml
        #    are available, then a prompt resolved from the filename via those
        #    config files is used. (this is initially resolved in
        #    ManuscriptEditor.revise_manuscript() and passed down to here via
        #    the 'resolved_prompt' argument.)
        # 3. if a section_name is specified, then a canned section-specific
        #    prompt matching the section name is used.
        # 4. finally, if none of the above are true, then a generic prompt is
        #    used.

        # set of options to replace in the prompt text, e.g.
        # {title} would be replaced with self.title, the title of
        # the manuscript.
        placeholders = {
            "paragraph_text": paragraph_text.strip(),
            "section_name": section_name,
            "title": self.title,
            "keywords": ", ".join(self.keywords),
        }

        custom_prompt = None
        if (c := os.environ.get(env_vars.CUSTOM_PROMPT, "").strip()) and c != "":
            custom_prompt = c
            print(
                f"Using custom prompt from environment variable '{env_vars.CUSTOM_PROMPT}'"
            )

            prompt = custom_prompt.format(**placeholders)
        elif resolved_prompt:
            # use the resolved prompt from the ai-revision config files, if available
            # replace placeholders with their actual values
            prompt = resolved_prompt.format(**placeholders)
        elif section_name in ("abstract",):
            prompt = f"""
                Revise the following paragraph from the {section_name} of an academic paper (with the title '{self.title}' and keywords '{", ".join(self.keywords)}')
                so the research problem/question is clear,
                   the solution proposed is clear,
                   the text grammar is correct, spelling errors are fixed,
                   and the text is in active voice and has a clear sentence structure
            """
        elif section_name in ("introduction", "discussion"):
            prompt = f"""
                Revise the following paragraph from the {section_name.capitalize()} section of an academic paper (with the title '{self.title}' and keywords '{", ".join(self.keywords)}')
                so
                   most of the citations to other academic papers are kept,
                   the text minimizes the use of jargon,
                   the text grammar is correct, spelling errors are fixed,
                   and the text has a clear sentence structure
            """
        elif section_name in ("results",):
            prompt = f"""
                Revise the following paragraph from the {section_name.capitalize()} section of an academic paper (with the title '{self.title}' and keywords '{", ".join(self.keywords)}')
                so
                   most references to figures and tables are kept,
                   the details are enough to clearly explain the outcomes,
                   sentences are concise and to the point,
                   the text minimizes the use of jargon,
                   the text grammar is correct, spelling errors are fixed,
                   and the text has a clear sentence structure
            """
        elif section_name in ("methods",):
            equation_definition = r"$$ ... $$ {#id}"
            revise_sentence = f"""
                Revise the paragraph(s) below from
                the {section_name.capitalize()} section of an academic paper
                (with the title '{self.title}' and keywords '{", ".join(self.keywords)}')
            """.strip()

            prompt = f"""
                {revise_sentence}
                so
                   most of the citations to other academic papers are kept,
                   most of the technical details are kept,
                   most references to equations (such as "Equation (@id)") are kept,
                   all equations definitions (such as '{equation_definition}') are included with newlines before and after,
                   the most important symbols in equations are defined,
                   spelling errors are fixed, the text grammar is correct,
                   and the text has a clear sentence structure
            """.strip()
        else:
            prompt = "Revise the following paragraph"

            if section_name is not None and section_name != "":
                prompt += f" from the {section_name.capitalize()} section"

            prompt += f" of an academic paper (with title '{self.title}' and keywords '{', '.join(self.keywords)}')"
            prompt += """
                so
                    the text minimizes the use of jargon,
                    the text grammar is correct, spelling errors are fixed,
                    and the text has a clear sentence structure
            """

        # replace multiple spaces with a single space only if there is no custom prompt,
        # since otherwise the custom prompt might have the paragraph text within, and
        # we are not supposed to reformat that here.
        if custom_prompt is None:
            prompt = self.several_spaces_pattern.sub(" ", prompt).strip()

        if custom_prompt is not None and "{paragraph_text}" in custom_prompt:
            return prompt

        return f"{prompt}.\n\n{paragraph_text.strip()}"

    def get_max_tokens(self, paragraph_text: str, fraction: float = 2.0) -> int:
        """
        Returns the maximum number of tokens that can be generated by the model.
        It uses a fration of the total number of tokens in the paragraph to
        avoid the model to generate too many tokens.

        If the environment variable
        name given by env_vars.MAX_TOKENS_PER_REQUEST is set, then it will be used
        instead of the fraction. If the value of the environment variable
        contains a dot (such as 0.5 or 2.7) it will be interpreted as a fraction
        of the total number of tokens in the paragraph. Otherwise, it will be
        interpreted as an absolute number of tokens (such as 250).

        Args:
            paragraph_text: The text of the paragraph to be revised.
            fraction: The fraction of the total number of tokens in the
                paragraph.

        Returns:
            The maximum number of tokens that can be generated by the model.
        """
        # To estimate the number of tokens, we follow the rule of thumb that
        # "one token generally corresponds to ~4 characters of text for common
        # English text" (https://beta.openai.com/tokenizer)
        estimated_tokens_in_paragraph_text = int(len(paragraph_text) / 4)

        if env_vars.MAX_TOKENS_PER_REQUEST in os.environ:
            max_tokens_in_env = os.environ[env_vars.MAX_TOKENS_PER_REQUEST]
            if "." in max_tokens_in_env:
                fraction = float(max_tokens_in_env)
            else:
                return int(max_tokens_in_env)

        return int(estimated_tokens_in_paragraph_text * fraction)

    @staticmethod
    def get_max_tokens_from_error_message(error_message: str) -> dict[str, int] | None:
        # attempt to extract the "maximum context length", "requested tokens",
        # "tokens in the prompt" and "token in completion" from the error message
        # to adjust the query and retry
        max_context_length = re.search(
            r"maximum context length is (\d+)", error_message
        )
        if max_context_length is None:
            return
        max_context_length = int(max_context_length.group(1))

        requested_tokens = re.search(
            r"however you requested (\d+) tokens", error_message
        )
        if requested_tokens is None:
            return
        requested_tokens = int(requested_tokens.group(1))

        tokens_in_prompt = re.search(r"\((\d+) in your prompt;", error_message)
        if tokens_in_prompt is None:
            return
        tokens_in_prompt = int(tokens_in_prompt.group(1))

        tokens_in_completion = re.search(r"; (\d+) for the completion", error_message)
        if tokens_in_completion is None:
            return
        tokens_in_completion = int(tokens_in_completion.group(1))

        return {
            "max_context_length": max_context_length,
            "requested_tokens": requested_tokens,
            "tokens_in_prompt": tokens_in_prompt,
            "tokens_in_completion": tokens_in_completion,
        }

    def get_params(self, paragraph_text, section_name, resolved_prompt=None):
        """
        Given the paragraph text and section name, produces parameters that are
        used when invoking an LLM via an API.

        The specific parameters vary depending on the endpoint being used, which
        is determined by the model that was chosen when GPT3CompletionModel was
        instantiated.

        Args:
            paragraph_text: The text of the paragraph to be revised.
            section_name: The name of the section the paragraph belongs to.
            resolved_prompt: The prompt resolved via ai-revision config files, if available.

        Returns:
            A dictionary of parameters to be used when invoking an LLM API.
        """
        max_tokens = self.get_max_tokens(paragraph_text)
        prompt = self.get_prompt(paragraph_text, section_name, resolved_prompt)

        params = {
            "n": 1,
        }

        if self.endpoint == "chat":
            params.update(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "stop": None,
                }
            )
        else:
            params.update(
                {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "stop": None,
                }
            )

        params.update(self.model_parameters)

        return params

    def revise_paragraph(
        self, paragraph_text: str, section_name: str = None, resolved_prompt=None
    ):
        """
        It revises a paragraph using GPT-3 completion model.

        Arguments:
            paragraph_text (str): Paragraph text to revise.
            section_name (str): Section name of the paragrap
            resolved_prompt (str): Prompt resolved via ai-revision config files, if available.

        Returns:
            Revised paragraph text.
        """

        # based on the paragraph text to revise and the section to which it
        # belongs, constructs parameters that we'll use to query the LLM's API
        params = self.get_params(paragraph_text, section_name, resolved_prompt)

        retry_count = 0
        message = ""
        while message == "" and retry_count < self.retry_count:
            try:
                print(
                    f"[Attempt #{retry_count}] Revising paragraph '{paragraph_text[:20]}'...",
                    flush=True,
                )

                # map the prompt to langchain's prompt types, based on what
                # kind of endpoint we're using
                if "messages" in params:
                    # map the messages to langchain's message types
                    # based on the 'role' field
                    prompt = [
                        (
                            HumanMessage(content=msg["content"])
                            if msg["role"] == "user"
                            else SystemMessage(content=msg["content"])
                        )
                        for msg in params["messages"]
                    ]
                elif "prompt" in params:
                    prompt = [HumanMessage(content=params["prompt"])]

                response = self.client.invoke(
                    input=prompt,
                    max_tokens=params.get("max_tokens"),
                    stop=params.get("stop"),
                )

                if isinstance(response, BaseMessage):
                    message = response.content.strip()
                else:
                    message = response.strip()

            except Exception as e:
                error_message = str(e)
                print(f"Error: {error_message}")

                # if the error message suggests to sample again, let's do that
                if "Please sample again" in error_message:
                    pass
                elif (
                    "invalid x-api-key" in error_message
                    or "invalid_api_key" in error_message
                ):
                    # raise an error if the API key is invalid.
                    # this is treated as a fatal error in ManuscriptEditor's
                    # revise_and_write_paragraph() method, rather than being
                    # reported as a warning in the emitted content as all other
                    # exceptions are.
                    raise APIKeyInvalidError(
                        f"Invalid API key used for provider '{self._model_provider}'"
                    ) from e
                elif "overloaded" in error_message:
                    time.sleep(5)
                elif (
                    "limit reached" in error_message
                    and "on requests per min" in error_message
                ):
                    # wait a little before retrying
                    time.sleep(30)
                else:
                    # if the error mesaage suggests to reduce the number of tokens,
                    # obtain the number of tokens to reduce and retry
                    token_stats = self.get_max_tokens_from_error_message(error_message)

                    if token_stats is None:
                        raise e

                    max_context_length = token_stats["max_context_length"]
                    tokens_in_prompt = token_stats["tokens_in_prompt"]

                    new_max_tokens = max_context_length - tokens_in_prompt

                    if new_max_tokens <= 0:
                        raise e

                    params["max_tokens"] = new_max_tokens
            finally:
                retry_count += 1

        return message


class DebuggingManuscriptRevisionModel(GPT3CompletionModel):
    """
    This model returns the same paragraph and important information submitted to
    the final revision function (i.e., that hits the remote API), such as the section
    name and the resolved prompt.
    """

    def __init__(self, *args, **kwargs):
        if "title" not in kwargs or kwargs["title"] is None:
            kwargs["title"] = "Debugging Title"
        if "keywords" not in kwargs or kwargs["keywords"] is None:
            kwargs["keywords"] = ["debugging", "keywords"]

        super().__init__(*args, **kwargs)

    def revise_paragraph(self, paragraph_text, section_name, resolved_prompt=None):
        params = self.get_params(paragraph_text, section_name, resolved_prompt)
        json_params = json.dumps(params, indent=4)
        return f"%%%PARAGRAPH START%%%\n{json_params}\n%%%PARAGRAPH END%%%"
