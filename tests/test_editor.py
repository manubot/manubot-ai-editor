from pathlib import Path

from chatgpt_editor.editor import ManuscriptEditor
from chatgpt_editor.models import GPT3CompletionModel

MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts"


def test_revise_abstract(tmp_path):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model = GPT3CompletionModel(
        title=me.title,
        keywords=me.keywords,
    )

    me.revise_file("01.abstract.md", tmp_path, model)
    pass

# test output dir that does not exists: it should be created first
# test with a dummy model first that returns the same paragraph, just to check that paragraphs are correctly parsed
