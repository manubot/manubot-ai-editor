"""
Tests basic functions of the models module that do not require access to an external API.
"""

import os
from pathlib import Path
from unittest import mock

import pytest

from manubot_ai_editor.editor import ManuscriptEditor, env_vars
from manubot_ai_editor import models
from manubot_ai_editor.models import GPT3CompletionModel, RandomManuscriptRevisionModel

MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts"


def test_model_object_init_without_openai_api_key():
    _environ = os.environ.copy()
    try:
        if env_vars.OPENAI_API_KEY in os.environ:
            os.environ.pop(env_vars.OPENAI_API_KEY)

        with pytest.raises(ValueError):
            GPT3CompletionModel(
                title="Test title",
                keywords=["test", "keywords"],
            )
    finally:
        os.environ = _environ


@mock.patch.dict("os.environ", {env_vars.OPENAI_API_KEY: "env_var_test_value"})
def test_model_object_init_with_openai_api_key_as_environment_variable():
    GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    assert models.openai.api_key == "env_var_test_value"


def test_model_object_init_with_openai_api_key_as_parameter():
    _environ = os.environ.copy()
    try:
        if env_vars.OPENAI_API_KEY in os.environ:
            os.environ.pop(env_vars.OPENAI_API_KEY)

        GPT3CompletionModel(
            title="Test title",
            keywords=["test", "keywords"],
            openai_api_key="test_value",
        )

        from manubot_ai_editor import models

        assert models.openai.api_key == "test_value"
    finally:
        os.environ = _environ


@mock.patch.dict("os.environ", {env_vars.OPENAI_API_KEY: "env_var_test_value"})
def test_model_object_init_with_openai_api_key_as_parameter_has_higher_priority():
    GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
        openai_api_key="test_value",
    )

    from manubot_ai_editor import models

    assert models.openai.api_key == "test_value"


def test_model_object_init_default_language_model():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    assert model.model_parameters["model"] == "gpt-3.5-turbo"


@mock.patch.dict("os.environ", {env_vars.LANGUAGE_MODEL: "text-curie-001"})
def test_model_object_init_read_language_model_from_environment():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    assert model.model_parameters["model"] == "text-curie-001"


@mock.patch.dict("os.environ", {env_vars.LANGUAGE_MODEL: ""})
def test_model_object_init_read_language_model_from_environment_is_empty():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    assert model.model_parameters["model"] == "gpt-3.5-turbo"


def test_get_max_tokens_fraction_is_one():
    paragraph = r"""
Correlation coefficients are widely used to identify patterns in data that may be of particular interest.
In transcriptomics, genes with correlated expression often share functions or are part of disease-relevant biological processes.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    paragraph_text = " ".join(paragraph)

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model = GPT3CompletionModel(
        title=me.title,
        keywords=me.keywords,
    )

    max_tokens = model.get_max_tokens(paragraph_text, 1.0)
    assert max_tokens is not None
    assert isinstance(max_tokens, int)
    assert 50 < max_tokens < 60


def test_get_max_tokens_using_fraction_is_two():
    paragraph = r"""
Correlation coefficients are widely used to identify patterns in data that may be of particular interest.
In transcriptomics, genes with correlated expression often share functions or are part of disease-relevant biological processes.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    paragraph_text = " ".join(paragraph)

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model = GPT3CompletionModel(
        title=me.title,
        keywords=me.keywords,
    )

    max_tokens = model.get_max_tokens(paragraph_text, 2.0)
    assert max_tokens is not None
    assert isinstance(max_tokens, int)
    assert 110 < max_tokens < 120


@mock.patch.dict("os.environ", {env_vars.MAX_TOKENS_PER_REQUEST: "0.5"})
def test_get_max_tokens_using_fraction_is_given_by_environment_and_is_float():
    paragraph = r"""
Correlation coefficients are widely used to identify patterns in data that may be of particular interest.
In transcriptomics, genes with correlated expression often share functions or are part of disease-relevant biological processes.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    paragraph_text = " ".join(paragraph)

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model = GPT3CompletionModel(
        title=me.title,
        keywords=me.keywords,
    )

    max_tokens = model.get_max_tokens(paragraph_text)
    assert max_tokens is not None
    assert isinstance(max_tokens, int)
    assert 25 < max_tokens < 35


@mock.patch.dict("os.environ", {env_vars.MAX_TOKENS_PER_REQUEST: "779"})
def test_get_max_tokens_using_fraction_is_given_by_environment_and_is_int():
    paragraph = r"""
Correlation coefficients are widely used to identify patterns in data that may be of particular interest.
In transcriptomics, genes with correlated expression often share functions or are part of disease-relevant biological processes.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    paragraph_text = " ".join(paragraph)

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model = GPT3CompletionModel(
        title=me.title,
        keywords=me.keywords,
    )

    # parameter 'fraction' is ignored when environment variable is set
    max_tokens = model.get_max_tokens(paragraph_text, 2.0)
    assert max_tokens is not None
    assert isinstance(max_tokens, int)
    assert max_tokens == 779


def test_revise_paragraph_too_few_sentences():
    # from LLM for articles revision manuscript
    paragraph = r"""
Since the gold standard of drug-disease medical indications is described with Disease Ontology IDs (DOID) [@doi:10.1093/nar/gky1032], we mapped PhenomeXcan traits to the Experimental Factor Ontology [@doi:10.1093/bioinformatics/btq099] using [@url:https://github.com/EBISPOT/EFO-UKB-mappings], and then to DOID.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 1

    model = RandomManuscriptRevisionModel()

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "methods"
    )
    assert paragraph_text is not None
    assert isinstance(paragraph_text, str)

    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised == paragraph_text
    assert len(paragraph_revised) > 10
    assert "<!--\nERROR:" not in paragraph_revised


def test_revise_paragraph_too_few_words():
    # from LLM for articles revision manuscript
    paragraph = r"""
We ran our regression model for all 987 LVs across the 4,091 traits in PhenomeXcan.
For replication, we ran the model in the 309 phecodes in eMERGE.
We adjusted the $p$-values using the Benjamini-Hochberg procedure.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 3

    model = RandomManuscriptRevisionModel()

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "methods"
    )
    assert paragraph_text is not None
    assert isinstance(paragraph_text, str)

    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised == paragraph_text
    assert len(paragraph_revised) > 10
    assert "<!--\nERROR:" not in paragraph_revised
