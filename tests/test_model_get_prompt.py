from unittest import mock

from manubot_ai_editor import env_vars
from manubot_ai_editor.models import GPT3CompletionModel, DEFAULT_PROMPT_OVERRIDE
import pytest

# test decorator to skip section-specific prompt tests when
# DEFAULT_PROMPT_OVERRIDE is set
default_prompt_override = pytest.mark.skipif(
    DEFAULT_PROMPT_OVERRIDE is not None, reason="DEFAULT_PROMPT_OVERRIDE is set, so section-specific prompt tests are off"
)

# test patcher that sets DEFAULT_PROMPT_OVERRIDE to None, so that the
# tests that are invalidated by it being set can still run
reset_default_prompt_override = mock.patch("manubot_ai_editor.models.DEFAULT_PROMPT_OVERRIDE", None)

# ==============================
# === default prompt override test
# ==============================

@pytest.mark.skipif(
    DEFAULT_PROMPT_OVERRIDE is None,
    reason="DEFAULT_PROMPT_OVERRIDE is not set, so this test is not needed"
)
def test_get_prompt_default_prompt_override():
    """"
    Tests that, when DEFAULT_PROMPT_OVERRIDE is set, the prompt matches
    the default prompt rather than section-specific prompts.

    Note that we test for the explicit text "Proofread the following paragraph..."
    rather than just checking that it's equal to DEFAULT_PROMPT_OVERRIDE, since
    we also want to validate that it's the default prompt MP specified.
    """
    
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert prompt.startswith("Proofread the following paragraph that is part of a scientific manuscript.")
    assert prompt.endswith(paragraph_text[-20:])
    assert "  " not in prompt


# ==============================
# === section-specific prompt tests
# ==============================

# since these all rely on the section-specific default prompt resolver, which is
# disabled when DEFAULT_PROMPT_OVERRIDE is set, we should skip these tests.

@reset_default_prompt_override
def test_get_prompt_for_abstract():
    manuscript_title = "Title of the manuscript to be revised"
    manuscript_keywords = ["keyword0", "keyword1", "keyword2"]

    model = GPT3CompletionModel(
        title=manuscript_title,
        keywords=manuscript_keywords,
    )

    paragraph_text = "Text of the abstract"

    prompt = model.get_prompt(paragraph_text, "abstract")
    assert prompt is not None
    assert isinstance(prompt, str)
    assert "abstract" in prompt
    assert f"'{manuscript_title}'" in prompt
    assert f"{manuscript_keywords[0]}" in prompt
    assert f"{manuscript_keywords[1]}" in prompt
    assert f"{manuscript_keywords[2]}" in prompt
    assert paragraph_text in prompt
    assert prompt.startswith("Revise")
    assert "  " not in prompt


@reset_default_prompt_override
def test_get_prompt_for_abstract_edit_endpoint():
    manuscript_title = "Title of the manuscript to be revised"
    manuscript_keywords = ["keyword0", "keyword1", "keyword2"]

    model = GPT3CompletionModel(
        title=manuscript_title,
        keywords=manuscript_keywords,
        model_engine="text-davinci-edit-001",
    )

    paragraph_text = "Text of the abstract. "

    instruction, paragraph = model.get_prompt(paragraph_text, "abstract")
    assert instruction is not None
    assert isinstance(instruction, str)
    assert paragraph is not None
    assert isinstance(paragraph, str)

    assert "this paragraph" in instruction
    assert "abstract" in instruction
    assert f"'{manuscript_title}'" in instruction
    assert f"{manuscript_keywords[0]}" in instruction
    assert f"{manuscript_keywords[1]}" in instruction
    assert f"{manuscript_keywords[2]}" in instruction
    assert "  " not in instruction
    assert instruction.startswith("Revise")

    assert paragraph_text.strip() == paragraph


@reset_default_prompt_override
def test_get_prompt_for_introduction():
    manuscript_title = "Title of the manuscript to be revised"
    manuscript_keywords = ["keyword0", "keyword1", "keyword2"]

    model = GPT3CompletionModel(
        title=manuscript_title,
        keywords=manuscript_keywords,
    )

    paragraph_text = "Text of the initial part"

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert prompt is not None
    assert isinstance(prompt, str)
    assert "Introduction" in prompt
    assert f"'{manuscript_title}'" in prompt
    assert f"{manuscript_keywords[0]}" in prompt
    assert f"{manuscript_keywords[1]}" in prompt
    assert f"{manuscript_keywords[2]}" in prompt
    assert paragraph_text in prompt
    assert prompt.startswith("Revise")
    assert "  " not in prompt


@reset_default_prompt_override
def test_get_prompt_section_is_abstract():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "abstract")
    assert prompt.startswith(
        "Revise the following paragraph from the abstract of an academic paper "
    )
    assert prompt.endswith(paragraph_text[-20:])


@reset_default_prompt_override
def test_get_prompt_section_is_introduction():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert prompt.startswith(
        "Revise the following paragraph from the Introduction section "
    )
    assert prompt.endswith(paragraph_text[-20:])


@reset_default_prompt_override
def test_get_prompt_section_is_discussion():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "discussion")
    assert prompt.startswith(
        "Revise the following paragraph from the Discussion section "
    )
    assert prompt.endswith(paragraph_text[-20:])


@reset_default_prompt_override
def test_get_prompt_section_is_methods():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "methods")
    assert prompt.startswith("Revise the paragraph(s) below from the Methods ")
    assert prompt.endswith(paragraph_text[-20:])


@reset_default_prompt_override
def test_get_prompt_section_is_results():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "results")
    assert prompt.startswith("Revise the following paragraph from the Results section ")
    assert prompt.endswith(paragraph_text[-20:])


@reset_default_prompt_override
def test_get_prompt_not_standard_section():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "acknowledgements")
    assert prompt.startswith(
        "Revise the following paragraph from the Acknowledgements section of an academic paper "
        "(with title 'Test title' and keywords 'test, keywords') so the text "
    )
    assert prompt.endswith(paragraph_text[-20:])


@reset_default_prompt_override
def test_get_prompt_section_not_provided():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text)
    assert prompt.startswith(
        "Revise the following paragraph of an academic paper "
        "(with title 'Test title' and keywords 'test, keywords') so the text "
    )
    assert prompt.endswith(paragraph_text[-20:])


@reset_default_prompt_override
def test_get_prompt_section_is_none():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text)
    assert prompt.startswith("Revise the following paragraph of an academic paper ")
    assert prompt.endswith(paragraph_text[-20:])


# ==============================
# === custom prompt tests
# ==============================

@mock.patch.dict(
    "os.environ",
    {env_vars.CUSTOM_PROMPT: "proofread and revise the following paragraph"},
)
def test_get_prompt_custom_prompt_no_placeholders():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert (
        prompt == f"proofread and revise the following paragraph.\n\n{paragraph_text}"
    )

    prompt = model.get_prompt(paragraph_text, "methods")
    assert (
        prompt == f"proofread and revise the following paragraph.\n\n{paragraph_text}"
    )


@mock.patch.dict(
    "os.environ",
    {
        env_vars.CUSTOM_PROMPT: "proofread and revise the following paragraph from the {section_name} section"
    },
)
def test_get_prompt_custom_prompt_with_section_name():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert (
        prompt
        == f"proofread and revise the following paragraph from the introduction section.\n\n{paragraph_text}"
    )


@mock.patch.dict(
    "os.environ",
    {
        env_vars.CUSTOM_PROMPT: "proofread and revise the following paragraph of a manuscript with title '{title}'"
    },
)
def test_get_prompt_custom_prompt_with_manuscript_title():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert (
        prompt
        == f"proofread and revise the following paragraph of a manuscript with title 'Test title'.\n\n{paragraph_text}"
    )


@mock.patch.dict(
    "os.environ",
    {
        env_vars.CUSTOM_PROMPT: "proofread and revise the following paragraph of a manuscript with title '{title}' and keywords '{keywords}'"
    },
)
def test_get_prompt_custom_prompt_with_manuscript_title_and_keywords():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert (
        prompt
        == f"proofread and revise the following paragraph of a manuscript with title 'Test title' and keywords 'test, keywords'.\n\n{paragraph_text}"
    )


@mock.patch.dict(
    "os.environ",
    {
        env_vars.CUSTOM_PROMPT: "proofread and revise the following paragraph from the {section_name} section: {paragraph_text}"
    },
)
def test_get_prompt_custom_prompt_with_paragraph_text():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    paragraph_text = """
This is the first sentence.
And this is the second sentence.
Finally, the third sentence.
    """.strip()

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert (
        prompt
        == f"proofread and revise the following paragraph from the introduction section: {paragraph_text}"
    )
