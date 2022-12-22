from pathlib import Path

import pytest

from chatgpt_editor.editor import ManuscriptEditor
from chatgpt_editor.models import DummyManuscriptRevisionModel, GPT3CompletionModel

MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts"


def _check_nonparagraph_lines_are_preserved(input_filepath, output_filepath):
    """
    Checks whether non-paragraph lines in the input file are preserved in the output file.
    Non-paragraph lines are those that represent section headers, comments, blank lines, as
    defined in function ManuscriptEditor.line_is_not_part_of_paragraph.
    """
    # read lines from input file
    filepath = input_filepath
    assert filepath.exists()
    with open(filepath, "r") as infile:
        input_nonpar_lines = [
            line.rstrip()
            for line in infile
            if ManuscriptEditor.line_is_not_part_of_paragraph(line)
        ]

    # read lines from output file
    filepath = output_filepath
    assert filepath.exists()
    with open(filepath, "r") as infile:
        output_nonpar_lines = [
            line.rstrip()
            for line in infile
            if ManuscriptEditor.line_is_not_part_of_paragraph(line)
        ]

    # make sure all lines that are not "paragraphs" are preserved
    for input_line in input_nonpar_lines:
        # make sure there are nonparagraph lines left in output file
        assert (
            len(output_nonpar_lines) > 0
        ), "Output file has less non-paragraph lines than input file"

        # make sure nonparagraph lines are in the same order
        assert (
            input_line == output_nonpar_lines[0]
        ), f"{input_line} != {output_nonpar_lines[0]}"

        output_nonpar_lines.remove(input_line)

    # if nonparagraph lines were preserved, then the output_nonpar_liens should be empty
    assert (
        len(output_nonpar_lines) == 0
    ), f"Non-paragraph lines are not the same: {output_nonpar_lines}"


@pytest.mark.parametrize(
    "model",
    [
        DummyManuscriptRevisionModel(),
        GPT3CompletionModel(None, None),
    ],
)
def test_revise_abstract(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("01.abstract.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "01.abstract.md",
        output_filepath=tmp_path / "01.abstract.md",
    )


@pytest.mark.parametrize(
    "model",
    [
        DummyManuscriptRevisionModel(),
        GPT3CompletionModel(None, None),
    ],
)
def test_revise_introduction(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("02.introduction.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "02.introduction.md",
        output_filepath=tmp_path / "02.introduction.md",
    )


def test_get_section_from_filename():
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    assert me.get_section_from_filename("00.front-matter.md") is None
    assert me.get_section_from_filename("01.abstract.md") == "abstract"
    assert me.get_section_from_filename("02.introduction.md") == "introduction"
    assert me.get_section_from_filename("03.results.md") == "results"
    assert me.get_section_from_filename("04.discussion.md") == "discussion"
    assert me.get_section_from_filename("07.references.md") is None
    assert me.get_section_from_filename("06.acknowledgements.md") is None
    assert me.get_section_from_filename("08.supplementary.md") is None


@pytest.mark.parametrize(
    "model",
    [
        DummyManuscriptRevisionModel(),
        GPT3CompletionModel(None, None),
    ],
)
def test_revise_results_with_header_only(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("04.00.results.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "04.00.results.md",
        output_filepath=tmp_path / "04.00.results.md",
    )


@pytest.mark.parametrize(
    "model",
    [
        DummyManuscriptRevisionModel(),
        GPT3CompletionModel(None, None),
    ],
)
def test_revise_results_intro_with_figure(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("04.05.results_intro.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "04.05.results_intro.md",
        output_filepath=tmp_path / "04.05.results_intro.md",
    )

    # make sure the "image paragraph" was exactly copied to the output file
    assert (
        """
![
**Different types of relationships in data.**
Each panel contains a set of simulated data points described by two generic variables: $x$ and $y$.
The first row shows Anscombe's quartet with four different datasets (from Anscombe I to IV) and 11 data points each.
The second row contains a set of general patterns with 100 data points each.
Each panel shows the correlation value using Pearson ($p$), Spearman ($s$) and CCC ($c$).
Vertical and horizontal red lines show how CCC clustered data points using $x$ and $y$.
](images/intro/relationships.svg "Different types of relationships in data"){#fig:datasets_rel width="100%"}
    """.strip()
        in open(tmp_path / "04.05.results_intro.md").read()
    )

