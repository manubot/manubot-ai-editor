from pathlib import Path
from unittest import mock

from manubot_ai_editor.editor import ManuscriptEditor
from manubot_ai_editor.models import (
    GPT3CompletionModel,
    RandomManuscriptRevisionModel,
    DebuggingManuscriptRevisionModel
)
from manubot_ai_editor.prompt_config import IGNORE_FILE
import pytest

from utils.dir_union import mock_unify_open

MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts" / "phenoplier_full" / "content"
MANUSCRIPTS_CONFIG_DIR = Path(__file__).parent / "manuscripts" / "phenoplier_full" / "ci"


# check that this path exists and resolve it
def test_manuscripts_dir_exists():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    assert content_dir.exists()


# check that we can create a ManuscriptEditor object
def test_create_manuscript_editor():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    config_dir = MANUSCRIPTS_CONFIG_DIR.resolve(strict=True)
    editor = ManuscriptEditor(content_dir, config_dir)
    assert isinstance(editor, ManuscriptEditor)


# ==============================================================================
# === prompts tests, using ai-revision-config.yaml + ai-revision-prompts.yaml
# ==============================================================================

# contains standard prompt, config files for phenoplier_full
# (this is merged into the manuscript folder using the mock_unify_open mock)
PHENOPLIER_PROMPTS_DIR = (
    Path(__file__).parent / "config_loader_fixtures" / "phenoplier_full"
)


# check that we can resolve a file to a prompt, and that it's the correct prompt
@mock.patch("builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, PHENOPLIER_PROMPTS_DIR))
def test_resolve_prompt():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    config_dir = MANUSCRIPTS_CONFIG_DIR.resolve(strict=True)
    editor = ManuscriptEditor(content_dir, config_dir)

    phenoplier_files_matches = {
        # explicitly ignored in ai-revision-config.yaml
        "00.front-matter.md": (IGNORE_FILE, "front-matter"),
        # prompts that match a part of the filename
        "01.abstract.md": ("Test match abstract.\n", "abstract"),
        "02.introduction.md": (
            "Test match introduction or discussion.\n",
            "introduction",
        ),
        # these all match the regex 04\..+\.md, hence why the match object includes a suffix
        "04.00.results.md": ("Test match results.\n", "04.00.results.md"),
        "04.05.00.results_framework.md": (
            "Test match results.\n",
            "04.05.00.results_framework.md",
        ),
        "04.05.01.crispr.md": ("Test match results.\n", "04.05.01.crispr.md"),
        "04.15.drug_disease_prediction.md": (
            "Test match results.\n",
            "04.15.drug_disease_prediction.md",
        ),
        "04.20.00.traits_clustering.md": (
            "Test match results.\n",
            "04.20.00.traits_clustering.md",
        ),
        # more prompts that match a part of the filename
        "05.discussion.md": ("Test match introduction or discussion.\n", "discussion"),
        "07.00.methods.md": ("Test match methods.\n", "methods"),
        # these are all explicitly ignored in ai-revision-config.yaml
        "10.references.md": (IGNORE_FILE, "references"),
        "15.acknowledgements.md": (IGNORE_FILE, "acknowledgements"),
        "50.00.supplementary_material.md": (IGNORE_FILE, "supplementary_material"),
    }

    for filename, (expected_prompt, expected_match) in phenoplier_files_matches.items():
        prompt, match = editor.prompt_config.get_prompt_for_filename(filename)

        if expected_prompt is None:
            assert prompt is None
        else:
            # we strip() here so that tests still pass, even if the user uses
            # newlines to separate blocks and isn't aware that the trailing
            # newline becomes part of the value
            assert prompt.strip() == expected_prompt.strip()

        if expected_match is None:
            assert match is None
        else:
            assert match.string[match.start() : match.end()] == expected_match


# test that we get the default prompt with a None match object for a
# file we don't recognize
@mock.patch("builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, PHENOPLIER_PROMPTS_DIR))
def test_resolve_default_prompt_unknown_file():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    config_dir = MANUSCRIPTS_CONFIG_DIR.resolve(strict=True)
    editor = ManuscriptEditor(content_dir, config_dir)

    prompt, match = editor.prompt_config.get_prompt_for_filename("some-unknown-file.md")

    assert prompt.strip() == """default prompt text"""
    assert match is None


# check that a file we don't recognize gets match==None and the 'default' prompt
# from the ai-revision-config.yaml file
@mock.patch("builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, PHENOPLIER_PROMPTS_DIR))
def test_unresolved_gets_default_prompt():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    config_dir = MANUSCRIPTS_CONFIG_DIR.resolve(strict=True)
    editor = ManuscriptEditor(content_dir, config_dir)
    prompt, match = editor.prompt_config.get_prompt_for_filename("crazy-filename")

    assert isinstance(prompt, str)
    assert match is None

    assert prompt.strip() == """default prompt text"""


# ==============================================================================
# === prompts_files tests, using ai-revision-prompts.yaml w/
# === ai-revision-config.yaml to process ignores, defaults
# ==============================================================================

# the following tests are derived from examples in
# https://github.com/manubot/manubot-ai-editor/issues/31
# we test four different scenarios from ./config_loader_fixtures:
# - Only ai-revision-prompts.yaml is defined (only_revision_prompts)
ONLY_REV_PROMPTS_DIR = (
    Path(__file__).parent / "config_loader_fixtures" / "only_revision_prompts"
)
# - Both ai-revision-prompts.yaml and ai-revision-config.yaml are defined (both_prompts_config)
BOTH_PROMPTS_CONFIG_DIR = (
    Path(__file__).parent / "config_loader_fixtures" / "both_prompts_config"
)
# - Only a single, generic prompt is defined (single_generic_prompt)
SINGLE_GENERIC_PROMPT_DIR = (
    Path(__file__).parent / "config_loader_fixtures" / "single_generic_prompt"
)
# - Both ai-revision-config.yaml and ai-revision-prompts.yaml specify filename matchings
#   (conflicting_promptsfiles_matchings)
CONFLICTING_PROMPTSFILES_MATCHINGS_DIR = (
    Path(__file__).parent / "config_loader_fixtures" / "conflicting_promptsfiles_matchings"
)
# ---
# test ManuscriptEditor.prompt_config sub-attributes are set correctly
# ---


def get_editor(content_dir=MANUSCRIPTS_DIR, config_dir=MANUSCRIPTS_CONFIG_DIR):
    content_dir = content_dir.resolve(strict=True)
    config_dir = config_dir.resolve(strict=True)
    editor = ManuscriptEditor(content_dir, config_dir)
    assert isinstance(editor, ManuscriptEditor)
    return editor


def test_no_config_unloaded():
    """
    With no config files defined, the ManuscriptPromptConfig object should
    have its attributes set to None.
    """
    editor = get_editor()

    # ensure that only the prompts defined in ai-revision-prompts.yaml are loaded
    assert editor.prompt_config.prompts is None
    assert editor.prompt_config.prompts_files is None
    assert editor.prompt_config.config is None


@mock.patch("builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, ONLY_REV_PROMPTS_DIR))
def test_only_rev_prompts_loaded():
    editor = get_editor()

    # ensure that only the prompts defined in ai-revision-prompts.yaml are loaded
    assert editor.prompt_config.prompts is None
    assert editor.prompt_config.prompts_files is not None
    assert editor.prompt_config.config is None


@mock.patch("builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, BOTH_PROMPTS_CONFIG_DIR))
def test_both_prompts_loaded():
    editor = get_editor()

    # ensure that only the prompts defined in ai-revision-prompts.yaml are loaded
    assert editor.prompt_config.prompts is not None
    assert editor.prompt_config.prompts_files is None
    assert editor.prompt_config.config is not None


@mock.patch(
    "builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, SINGLE_GENERIC_PROMPT_DIR)
)
def test_single_generic_loaded():
    editor = get_editor()

    # ensure that only the prompts defined in ai-revision-prompts.yaml are loaded
    assert editor.prompt_config.prompts is None
    assert editor.prompt_config.prompts_files is not None
    assert editor.prompt_config.config is not None


@mock.patch(
    "builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, CONFLICTING_PROMPTSFILES_MATCHINGS_DIR)
)
def test_conflicting_sources_warning(capfd):
    """
    Tests that a warning is printed when both ai-revision-prompts.yaml and
    ai-revision-config.yaml specify filename-to-prompt mappings.

    Specifically, the dicts that map filenames to prompts are:
    - ai-revision-prompts.yaml: 'prompts_files'
    - ai-revision-config.yaml: 'files.matchings'

    If both are specified, the 'files.matchings' key in ai-revision-config.yaml
    takes precedence, but a warning is printed.
    """

    editor = get_editor()

    # ensure that only the prompts defined in ai-revision-prompts.yaml are loaded
    assert editor.prompt_config.prompts is None
    assert editor.prompt_config.config is not None
    # for this test, we define both prompts_files and files.matchings which
    # creates a conflict that produces the warning we're looking for
    assert editor.prompt_config.prompts_files is not None
    assert editor.prompt_config.config['files']['matchings'] is not None

    expected_warning = (
        "WARNING: Both 'ai-revision-config.yaml' and "
        "'ai-revision-prompts.yaml' specify filename-to-prompt mappings. Only the "
        "'ai-revision-config.yaml' file's file.matchings section will be used; "
        "prompts_files will be ignored."
    )

    out, _ = capfd.readouterr()
    assert expected_warning in out


# ==============================================================================
# === test that ignored files are ignored in applicable scenarios
# ==============================================================================

# places in configs where files can be ignored:
# ai-revision-config.yaml: the `files.ignore` key
# ai-revision-prompts.yaml: when a prompt in `prompts_files` has a value of null


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        DebuggingManuscriptRevisionModel(
            title="Test title", keywords=["test", "keywords"]
        )
        # GPT3CompletionModel(None, None),
    ],
)
@mock.patch("builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, BOTH_PROMPTS_CONFIG_DIR))
def test_revise_entire_manuscript(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")
    me = get_editor()

    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    # after processing ignores, we should be left with 9 files from the original 12
    output_md_files = list(output_folder.glob("*.md"))
    assert len(output_md_files) == 9


@mock.patch("builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, BOTH_PROMPTS_CONFIG_DIR))
def test_revise_entire_manuscript_includes_title_keywords(tmp_path):
    from os.path import basename

    print(f"\n{str(tmp_path)}\n")
    me = get_editor()

    model = DebuggingManuscriptRevisionModel(
        title="Test title", keywords=["test", "keywords"]
    )

    # ensure overwriting the title and keywords works
    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    # gather up the output files so we can check their contents
    output_md_files = list(output_folder.glob("*.md"))

    # check that the title and keywords are in the final result
    # for prompts that include that information
    for output_md_file in output_md_files:
        # we expressly skip results because it doesn't contain any revisable
        # paragraphs
        if "results" in output_md_file.name:
            continue

        with open(output_md_file, "r") as f:
            content = f.read()
            assert me.title in content, f"not found in filename: {basename(output_md_file)}"
            assert ", ".join(me.keywords) in content, f"not found in filename: {basename(output_md_file)}"


# ==============================================================================
# === end-to-end tests, to verify that the prompts are making it into the final result
# ==============================================================================

PROMPT_PROPOGATION_CONFIG_DIR = (
    Path(__file__).parent / "config_loader_fixtures" / "prompt_propogation"
)

@mock.patch("builtins.open", mock_unify_open(MANUSCRIPTS_CONFIG_DIR, PROMPT_PROPOGATION_CONFIG_DIR))
def test_prompts_in_final_result(tmp_path):
    """
    Tests that the prompts are making it into the final resulting .md files.

    This test uses the DebuggingManuscriptRevisionModel, which is a model that
    inserts the prompt and other parameters into the final result. Using this
    model, we can test that the prompt we entered is used when applying the LLM.

    Note that 04.00.results.md contains no actual text, just a comment, so
    there's no paragraphs to assign a prompt and thus no result; we explicitly
    ignore the file in the config and in the test below.

    10.references.md also contains no actual text, just an HTML element where
    the references get inserted by another system (assumedly manubot), so we
    ignore it in the config and in this test as well.
    """
    me = get_editor()

    model = DebuggingManuscriptRevisionModel(
        title=me.title, keywords=me.keywords
    )

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    # mapping of filenames to prompts to check in the result
    files_to_prompts = {
        "00.front-matter.md": "This is the front-matter prompt.",
        "01.abstract.md": "This is the abstract prompt",
        "02.introduction.md": "This is the introduction prompt for the paper titled '%s'." % me.title,
        # "04.00.results.md": "This is the results prompt",
        "04.05.00.results_framework.md": "This is the results_framework prompt",
        "04.05.01.crispr.md": "This is the crispr prompt",
        "04.15.drug_disease_prediction.md": "This is the drug_disease_prediction prompt",
        "04.20.00.traits_clustering.md": "This is the traits_clustering prompt",
        "05.discussion.md": "This is the discussion prompt",
        "07.00.methods.md": "This is the methods prompt",
        # "10.references.md": "This is the references prompt",
        "15.acknowledgements.md": "This is the acknowledgements prompt",
        "50.00.supplementary_material.md": "This is the supplementary_material prompt",
    }

    # check that the prompts are in the final result
    output_md_files = list(output_folder.glob("*.md"))

    for output_md_file in output_md_files:
        with open(output_md_file, "r") as f:
            content = f.read()
            assert files_to_prompts[output_md_file.name].strip() in content


# ---------
# --- live GPT version of the test, with a different prompt
# ---------

# to save on time/cost, we use a version of the phenoplier manuscript that only
# contains the first paragraph of each section
BRIEF_MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts" / "phenoplier_full_only_first_para" / "content"
BRIEF_MANUSCRIPTS_CONFIG_DIR = Path(__file__).parent / "manuscripts" / "phenoplier_full_only_first_para" / "ci"

PROMPT_PROPOGATION_CONFIG_DIR = (
    Path(__file__).parent / "config_loader_fixtures" / "prompt_gpt3_e2e"
)

@pytest.mark.cost
@mock.patch("builtins.open", mock_unify_open(BRIEF_MANUSCRIPTS_CONFIG_DIR, PROMPT_PROPOGATION_CONFIG_DIR))
def test_prompts_apply_gpt3(tmp_path):
    """
    Tests that the custom prompts are applied when actually applying
    the prompts to an LLM.

    This test uses the GPT3CompletionModel, which performs a query againts
    the live OpenAI service, thus it does incur cost. Because of that,
    this test is marked 'cost' and requires the --runcost argument to be run,
    e.g. to run just this test: `pytest --runcost -k test_prompts_apply_gpt3`.

    As with test_prompts_in_final_result above, files that have no input and 
    thus no applied prompt are ignored.
    """
    me = get_editor(content_dir=BRIEF_MANUSCRIPTS_DIR, config_dir=BRIEF_MANUSCRIPTS_CONFIG_DIR)

    model = GPT3CompletionModel(
        title=me.title,
        keywords=me.keywords
    )

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    # mapping of filenames to keywords, present in the prompt, to check in the
    # result. (these words were generated by https://randomwordgenerator.com/,
    # fyi, not chosen for any particular reason)
    files_to_keywords = {
        "00.front-matter.md": "testify",
        "01.abstract.md": "bottle",
        "02.introduction.md": "wound",
        # "04.00.results.md": "classroom",
        "04.05.00.results_framework.md": "secretary",
        "04.05.01.crispr.md": "army",
        "04.15.drug_disease_prediction.md": "breakdown",
        "04.20.00.traits_clustering.md": "siege",
        "05.discussion.md": "beer",
        "07.00.methods.md": "confront",
        # "10.references.md": "disability",
        "15.acknowledgements.md": "stitch",
        "50.00.supplementary_material.md": "waiter",
    }

    # check that the prompts are in the final result
    output_md_files = list(output_folder.glob("*.md"))

    for output_md_file in output_md_files:
        with open(output_md_file, "r") as f:
            content = f.read()
            assert files_to_keywords[output_md_file.name].strip() in content
