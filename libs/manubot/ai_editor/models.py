import os
import re
from abc import ABC, abstractmethod
import random
import time

import openai

from manubot.ai_editor import env_vars


class ManuscriptRevisionModel(ABC):
    """
    An abstract class for manuscript revision models.
    """

    def __init__(self):
        pass

    @abstractmethod
    def revise_paragraph(self, paragraph_text, section_name):
        """
        It revises a paragraph of a manuscript from a given section.

        Args:
            paragraph_text (str): text of the paragraph to revise.
            section_name (str): name of the section the paragraph belongs to.

        Returns:
            Revised paragraph text.
        """
        raise NotImplemented

    @abstractmethod
    def get_prompt(self, paragraph_text, section_name):
        """
        Returns the prompt to be used for the revision of a paragraph that
        belongs to a given section.
        """
        raise NotImplemented


class DummyManuscriptRevisionModel(ManuscriptRevisionModel):
    """
    This model does nothing, just returns the same paragraph content with
    sentences one after the other separated by a white space (no new lines).
    """

    def __init__(self, add_paragraph_marks=False):
        super().__init__()
        self.sentence_end_pattern = re.compile(r".\n")
        self.add_paragraph_marks = add_paragraph_marks

    def revise_paragraph(self, paragraph_text, section_name):
        if self.add_paragraph_marks:
            return (
                "%%% PARAGRAPH START %%%\n"
                + paragraph_text.strip()
                + "\n%%% PARAGRAPH END %%%"
            )

        return self.sentence_end_pattern.sub(". ", paragraph_text).strip()

    def get_prompt(self, paragraph_text, section_name):
        return paragraph_text


class VerboseManuscriptRevisionModel(DummyManuscriptRevisionModel):
    """
    This model returns the same paragraph and adds a header to it.
    """

    def __init__(self, revised_header: str = "Revised:"):
        super().__init__()
        self.revised_header = revised_header

    def revise_paragraph(self, paragraph_text, section_name):
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

    def revise_paragraph(self, paragraph_text: str, section_name: str) -> str:
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

    def get_prompt(self, paragraph_text, section_name):
        return paragraph_text


class GPT3CompletionModel(ManuscriptRevisionModel):
    """
    Revises a paragraphs using GPT-3 completion model. Most of the parameters
    (https://beta.openai.com/docs/api-reference/completions/create) of the model
    can be specified either by parameters or environment variables.
    """

    def __init__(
        self,
        title: str,
        keywords: list[str],
        openai_api_key: str = None,
        model_engine: str = "text-davinci-003",
        temperature: float = 0.5,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        best_of: int = None,
        top_p: float = None,
        retry_count: int = 5,
        edit_endpoint: bool = False,
    ):
        super().__init__()

        # make sure the OpenAI API key is set
        openai.api_key = openai_api_key

        if openai.api_key is None:
            openai.api_key = os.environ.get(env_vars.OPENAI_API_KEY, None)

            if openai.api_key is None:
                raise ValueError(
                    f"OpenAI API key not found. Please provide it as parameter "
                    f"or set it as an the environment variable "
                    f"{env_vars.OPENAI_API_KEY}"
                )

        if env_vars.LANGUAGE_MODEL in os.environ:
            val = os.environ[env_vars.LANGUAGE_MODEL]
            if val.strip() != "":
                model_engine = val
                print(
                    f"Using language model from environment variable '{env_vars.LANGUAGE_MODEL}'"
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
        self.keywords = keywords

        # adjust options if edits endpoint was selected
        self.edit_endpoint = edit_endpoint
        if self.edit_endpoint and model_engine == "text-davinci-003":
            model_engine = "text-davinci-edit-001"

        if model_engine == "text-davinci-edit-001":
            self.edit_endpoint = True

        print("Language model: ", model_engine)

        self.model_parameters = {
            "engine": model_engine,
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

    def get_prompt(
        self, paragraph_text: str, section_name: str
    ) -> str | tuple[str, str]:
        """
        Returns the prompt to be used for the revision of a paragraph that
        belongs to a given section. There are three types of prompts according
        to the section: Abstract, Introduction, Methods, and the rest (i.e.
        Results, Discussion, etc.).

        Args:
            paragraph_text: text of the paragraph to revise.
            section_name: name of the section the paragraph belongs to.

        Returns:
            If self.edit_endpoint is False, then returns a string with the prompt to be used by the model for the revision of the paragraph.
            It contains two paragraphs of text: the command for the model
            ("Revise...") and the paragraph to revise.

            If self.edit_endpoint is True, then returns a tuple with two strings:
             1) the instructions to be used by the model for the revision of the paragraph,
             2) the paragraph to revise.
        """
        if section_name in ("abstract",):
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
            prompt = f"""
                Revise the following paragraph from the {section_name.capitalize()} section of an academic paper (with the title '{self.title}' and keywords '{", ".join(self.keywords)}')
                so
                   the text minimizes the use of jargon,
                   the text grammar is correct, spelling errors are fixed,
                   and the text has a clear sentence structure
            """

        prompt = self.several_spaces_pattern.sub(" ", prompt).strip()

        if not self.edit_endpoint:
            return f"{prompt}.\n\n{paragraph_text.strip()}"
        else:
            prompt = prompt.replace("the following paragraph", "this paragraph")
            return f"{prompt}.", paragraph_text.strip()

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

    def revise_paragraph(self, paragraph_text, section_name):
        """
        It revises a paragraph using GPT-3 completion model.

        Arguments:
            paragraph_text (str): Paragraph text to revise.
            section_name (str): Section name of the paragraph.
            throw_error (bool): If True, it throws an error if the API call fails.
                If False, it returns the original paragraph text.

        Returns:
            Revised paragraph text.
        """
        max_tokens = self.get_max_tokens(paragraph_text)
        prompt = self.get_prompt(paragraph_text, section_name)

        params = {
            "n": 1,
        }

        if self.edit_endpoint:
            params.update(
                {
                    "instruction": prompt[0],
                    "input": prompt[1],
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

        retry_count = 0
        message = ""
        while message == "" and retry_count < self.retry_count:
            try:
                print(
                    f"[Attempt #{retry_count}] Revising paragraph '{paragraph_text[:20]}'...",
                    flush=True,
                )

                if self.edit_endpoint:
                    completions = openai.Edit.create(**params)
                else:
                    completions = openai.Completion.create(**params)

                message = completions.choices[0].text.strip()
            except Exception as e:
                error_message = str(e)
                print(f"Error: {error_message}")

                # if the error message suggests to sample again, let's do that
                if "Please sample again" in error_message:
                    pass
                elif "overloaded" in error_message:
                    time.sleep(5)
                elif "limit reached" in error_message and "on requests per min" in error_message:
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

                    params["max_tokens"] = max_context_length - tokens_in_prompt
            finally:
                retry_count += 1

        return message
