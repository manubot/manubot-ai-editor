import os
import re
from abc import ABC, abstractmethod


class ManuscriptRevisionModel(ABC):
    """
    TODO: add docstring
    """

    def __init__(self):
        # Get ChatGPT API key and place into github secrets file
        assert (
            "OPENAI_API_KEY" in os.environ
        ), "OPENAI_API_KEY not found in environment variables"
        self.api_key = os.environ["OPENAI_API_KEY"]

    @abstractmethod
    def revise_paragraph(self, paragraph_text, section_name):
        """
        TODO: add docstring

        Args:
            paragraph_text (str):
            section_name (str):

        Returns:
            Revised paragraph text.
        """
        raise NotImplemented

    @abstractmethod
    def get_prompt(self, paragraph_text, section_name):
        """
        TODO: add docstring
        """
        raise NotImplemented


class DummyManuscriptRevisionModel(ManuscriptRevisionModel):
    """
    This model does nothing, just returns the same paragraph content with
    sentences one after the other using a white space (no new lines). This
    mimics what a real OpenAI model does.
    """

    def __init__(self):
        self.sentence_end_pattern = re.compile(r"\n")

    def revise_paragraph(self, paragraph_text, section_name):
        return self.sentence_end_pattern.sub(" ", paragraph_text).strip()

    def get_prompt(self, paragraph_text, section_name):
        return paragraph_text


class GPT3CompletionModel(ManuscriptRevisionModel):
    """
    Revises a paragraphs using GPT-3 completion model.

    TODO:
    - Read OpenAI API parameters: https://beta.openai.com/docs/api-reference/completions/create
      - There are many that are interesting, like temperature, top_p, presence_penalty, frequency_penalty, etc.
        "best_of" is interesting, but it can consume a lot of API calls.
      - Another interesting parameters is "user", it could be "manubot" here.
    """

    def __init__(
        self,
        title: str,
        keywords: list[str],
        model_engine: str = "text-davinci-003",
        temperature: float = 0.5,
    ):
        super().__init__()

        self.title = title
        self.keywords = keywords

        self.model_parameters = {
            "engine": model_engine,
            "temperature": temperature,
        }

        self.several_spaces_pattern = re.compile(r"\s+")

    def get_prompt(self, paragraph_text, section_name):
        if section_name in ("abstract",):
            prompt = f"""
                Revise the following {section_name} of an academic paper with title
                '{self.title}' and keywords '{", ".join(self.keywords)}', which
                is written in Markdown. Make sure the paragraph is easy to read,
                it is in active voice, and the take-home message is clear:
            """
        elif section_name in ("introduction",):
            prompt = f"""
                Revise the following paragraph of the {section_name} section of an
                academic paper with title '{self.title}' and keywords '{", ".join(self.keywords)}',
                which is written in Markdown. Make sure the paragraph is easy to read,
                it is in active voice, it has a clear and easy-to-read sentence structure,
                and it minimizes the use of jargon. Citations to other scientific articles
                are between square brackets and start with @doi, @pmid, etc., and should be kept:
            """
        elif section_name in ("results", "supplementary_material"):
            prompt = f"""
                Revise the following paragraph of the {section_name} section of an
                academic paper with title '{self.title}' and keywords '{", ".join(self.keywords)}',
                which is written in Markdown. Make sure the paragraph has a clear and easy-to-read sentence structure,
                and it minimizes the use of jargon. Figures must be always referenced at least once.
                Citations to other scientific articles are between square brackets and start with @doi, @pmid, etc.,
                and should be kept:
            """
        elif section_name in ("discussion",):
            prompt = f"""
                Revise the following paragraph of the {section_name} section of an
                academic paper with title '{self.title}' and keywords '{", ".join(self.keywords)}',
                which is written in Markdown. Make sure the paragraph has a clear and easy-to-read sentence structure,
                and it minimizes the use of jargon. Citations to other scientific articles
                are between square brackets and start with @doi, @pmid, etc., and should be kept:
            """
        elif section_name in ("methods",):
            prompt = f"""
                Revise the following paragraph of the {section_name} section of an
                academic paper with title '{self.title}' and keywords '{", ".join(self.keywords)}',
                which is written in Markdown. Make sure the paragraph has a clear and easy-to-read sentence structure,
                and it minimizes the use of jargon. Citations to other scientific articles
                are between square brackets and start with @doi, @pmid, etc., and must be kept.
                Formulas are between dollar signs ($) and must be kept:
            """
        else:
            raise ValueError(f"Section '{section_name}' not supported")

        prompt = self.several_spaces_pattern.sub(" ", prompt).strip()

        return f"{prompt}\n{paragraph_text.strip()}"

    def revise_paragraph(self, paragraph_text, section_name, throw_error=False):
        """
        It revises a paragraph using GPT-3 completion model.

        Arguments:
            paragraph_text (str): Paragraph text to revise.
            section_name (str): Section name of the paragraph.
            throw_error (bool): If True, it throws an error if the API call fails.
                If False, it returns the original paragraph text.

        Returns:
            Revised paragraph text.

        TODO:
          - Add reduction_fraction, between 0 and 1, which multiplies the paragraph
            length by this fraction before sending to GPT-3. This is useful to
            force summarization.
        """
        import openai

        paragraph_length = len(paragraph_text)
        prompt = self.get_prompt(paragraph_text, section_name)

        try:
            completions = openai.Completion.create(
                engine=self.model_parameters["engine"],
                prompt=prompt,
                max_tokens=paragraph_length,
                n=1,
                stop=None,
                temperature=self.model_parameters["temperature"],
            )
        except openai.error.InvalidRequestError as e:
            if throw_error:
                raise e
            else:
                return paragraph_text

        message = completions.choices[0].text
        return message.strip()
