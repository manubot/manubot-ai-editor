import os
import re
from abc import ABC, abstractmethod

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
    This mimics what a real OpenAI model does.
    """

    def __init__(self):
        super().__init__()
        self.sentence_end_pattern = re.compile(r"\n")

    def revise_paragraph(self, paragraph_text, section_name):
        return self.sentence_end_pattern.sub(" ", paragraph_text).strip()

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
            model_engine = os.environ[env_vars.LANGUAGE_MODEL]
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
                # if it is not a float, we ignore it
                pass

        self.title = title
        self.keywords = keywords

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

    def get_prompt(self, paragraph_text: str, section_name: str) -> str:
        """
        Returns the prompt to be used for the revision of a paragraph that belongs
        to a given section. There are three types of prompts according to the section:
        Abstract, Introduction and the rest (i.e. Methods, Results, Discussion, etc.).

        Args:
            paragraph_text: text of the paragraph to revise.
            section_name: name of the section the paragraph belongs to.

        Returns:
            Prompt to be used by the model for the revision of the paragraph.
            It contains two paragraphs of text: the command for the model
            ("Revise...") and the paragraph to revise.
        """
        if section_name in ("abstract",):
            prompt = f"""
                Revise the following paragraph (in Markdown format) of the {section_name.capitalize()} of an academic paper with title '{self.title}' and keywords '{", ".join(self.keywords)}'.
                Make sure the paragraph is easy to read, it is in active voice, and the take-home message is clear.
            """
        elif section_name in ("introduction",):
            prompt = f"""
                Revise the following paragraph (in Markdown format) of the {section_name.capitalize()} section of an academic paper with title '{self.title}' and keywords '{", ".join(self.keywords)}'.
                Make sure the paragraph has a clear and easy-to-read sentence structure, it is in active voice, and it minimizes the use of jargon.
                Keep most of the citations to other academic papers.
                Keep the Markdown formatting.
            """
        else:
            prompt = f"""
                Revise the following paragraph (in Markdown format) of the {section_name.capitalize()} section of an academic paper with title '{self.title}' and keywords '{", ".join(self.keywords)}'.
                Make sure the paragraph has a clear and easy-to-read sentence structure, and it minimizes the use of jargon.
                Keep the Markdown formatting.
            """

        prompt = self.several_spaces_pattern.sub(" ", prompt).strip()

        return f"{prompt}\n\n{paragraph_text.strip()}"

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
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stop": None,
            "n": 1,
        }

        params.update(self.model_parameters)

        completions = openai.Completion.create(**params)

        message = completions.choices[0].text
        return message.strip()
