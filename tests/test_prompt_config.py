
import os
import re
from pathlib import Path
from unittest import mock

import pytest

from manubot_ai_editor.editor import ManuscriptEditor, env_vars
from manubot_ai_editor import models
from manubot_ai_editor.models import GPT3CompletionModel, RandomManuscriptRevisionModel

MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts" / "phenoplier_full"

# check that this path exists and resolve it
def test_manuscripts_dir_exists():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    assert content_dir.exists()

# check that we can create a ManuscriptEditor object
def test_create_manuscript_editor():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    editor = ManuscriptEditor(content_dir)
    assert isinstance(editor, ManuscriptEditor)


# ==============================================================================
# === prompts tests, using ai_revision-config.yaml + ai_revision-prompts.yaml
# ==============================================================================

# check that we can resolve a file to a prompt, and that it's the correct prompt
def test_resolve_prompt():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    editor = ManuscriptEditor(content_dir)

    phenoplier_files_matches = {
        # explicitly ignored in ai_revision-config.yaml
        '00.front-matter.md': (None, 'front-matter'),

        # prompts that match a part of the filename
        '01.abstract.md': ('Test match abstract.\n', 'abstract'),
        '02.introduction.md': ('Test match introduction or discussion.\n', 'introduction'),

        # these all match the regex 04\..+\.md, hence why the match object includes a suffix
        '04.00.results.md': ('Test match results.\n', '04.00.results.md'),
        '04.05.00.results_framework.md': ('Test match results.\n', '04.05.00.results_framework.md'),
        '04.05.01.crispr.md': ('Test match results.\n', '04.05.01.crispr.md'),
        '04.15.drug_disease_prediction.md': ('Test match results.\n', '04.15.drug_disease_prediction.md'),
        '04.20.00.traits_clustering.md': ('Test match results.\n', '04.20.00.traits_clustering.md'),

        # more prompts that match a part of the filename
        '05.discussion.md': ('Test match introduction or discussion.\n', 'discussion'),
        '07.00.methods.md': ('Test match methods.\n', 'methods'),

        # these are all explicitly ignored in ai_revision-config.yaml
        '10.references.md': (None, 'references'),
        '15.acknowledgements.md': (None, 'acknowledgements'),
        '50.00.supplementary_material.md': (None, 'supplementary_material')
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
            assert match.string[match.start():match.end()] == expected_match

# test that we get the default prompt with a None match object for a
# file we don't recognize
def test_resolve_default_prompt_unknown_file():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    editor = ManuscriptEditor(content_dir)

    prompt, match = editor.prompt_config.get_prompt_for_filename("some-unknown-file.md")

    assert prompt.strip() == """default prompt text"""
    assert match is None

# check that a file we don't recognize gets match==None and the 'default' prompt
# from the ai_revision-config.yaml file
def test_unresolved_gets_default_prompt():
    content_dir = MANUSCRIPTS_DIR.resolve(strict=True)
    editor = ManuscriptEditor(content_dir)
    prompt, match = editor.prompt_config.get_prompt_for_filename("crazy-filename")

    assert isinstance(prompt, str)
    assert match is None

    assert prompt.strip() == """default prompt text"""

# ==============================================================================
# === prompts_files tests, using ai_revision-prompts.yaml w/ai_revision-config.yaml to process ignores, defaults
# ==============================================================================

# TBC