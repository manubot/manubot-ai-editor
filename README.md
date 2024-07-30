# Manubot AI Editor

A tool for performing automatic, AI-assisted revisions of [Manubot](https://manubot.org/) manuscripts.
Check out the [manuscript about this tool](https://greenelab.github.io/manubot-gpt-manuscript/) for more background information.

## Supported Large Language Models (LLMs)

We currently support OpenAI models only, and are working to add support for other models.
[Our evaluations](https://github.com/pivlab/manubot-ai-editor-evals) show that `gpt-4-turbo` is in general the best model for revising academic manuscripts.
Therefore, this is the default option.

## Using in a Manubot manuscript

Much of these instructions rely on the specific details of GitHub's website interface, which can change over time.
See their official docs for more info on [configuring GitHub Actions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository), [managing secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions#creating-secrets-for-a-repository), and [running workflows](https://docs.github.com/en/actions/using-workflows/manually-running-a-workflow).

### Setup

Start with a manuscript repo [forked from Manubot rootstock](https://github.com/manubot/rootstock), then follow these steps:

1. In your forks's "▶️ Actions" tab, enable GitHub Actions.
1. In your fork's "⚙️ Settings" tab, give GitHub Actions workflows read/write permissions and allow them to create pull requests.
1. If you haven't already, [make an OpenAI account](https://openai.com/api/) and [create an API key](https://platform.openai.com/api-keys).
1. In your fork's "⚙️ Settings" tab, make a new Actions repository secret with the name `OPENAI_API_KEY` and paste in your API key as the secret.

### Configuring prompts

In order to revise your manuscript, prompts must be provided to the AI model.
Manubot rootstock comes with several default, general-purpose prompts so that you can immediately use the AI editor without having to write and configure your own prompts.

But you can also define your own prompts, apply them to specific content, and control other behavior using YAML configuration files that you include with your manuscript.
See [docs/custom-prompts.md](https://github.com/manubot/manubot-ai-editor/blob/main/docs/custom-prompts.md) for more information.

### Running the editor

1. In your forks's "▶️ Actions" tab, go to the `ai-revision` workflow.
1. Manually run the workflow.
   You should see several options you can specify, such as the branch to revise and the AI model to use.
   [See these docs for an explanation of each option](https://github.com/manubot/manubot?tab=readme-ov-file#ai-assisted-academic-authoring).
1. Within a few minutes, the workflow should run, the editor should generate revisions, and a pull request should be created in your fork!

## Caveats

In the current implementation, the AI editor can only process, independently, one paragraph at a time.
This limits the contextual information the LLM receives and thus the specificity of what it can check and fix.
For instance, the revision process does not use information in other places of the manuscript to revise the current paragraph.
In addition, we provide section-specific prompts to revise text from different sections of the manuscript, such as the Abstract, Introduction, Results, etc.
However, some paragraphs from the same section [need different revision strategies](https://doi.org/10.1371/journal.pcbi.1005619).
For example, in the Discussion section of a manuscript, the first paragraph should typically summarize the findings from the Results section, while the rest of the paragraphs should follow a different structure.
The AI editor, however, can only judge each paragraph with the same section-specific prompt.

Finally, in addition to revising the paragraph using an LLM, the AI Editor will also perform some postprocessing of the revised text such as using one line per sentence to simplify diffs.
This might not work as expected in some cases.

We plan to reduce or remove these limitations in the future.

## Using from the command line

First, install Manubot in a Python environment, e.g.:

```bash
pip install --upgrade manubot[ai-rev]
```

You also need to export an environment variable with your OpenAI API key, e.g.:

```bash
export OPENAI_API_KEY=ABCD1234
```

You can also provide other environment variables that will change the behavior of the editor (such as revising certain files only).
For example, to specify the temperature parameter of OpenAI models, you can set the variable `export AI_EDITOR_TEMPERATURE=0.50`.
[See the complete list of supported variables](https://github.com/manubot/manubot-ai-editor/blob/main/libs/manubot_ai_editor/env_vars.py) documents.

Then, from the root directory of your Manubot manuscript, run the following:

```bash
# ⚠ THIS WILL OVERWRITE YOUR LOCAL MANUSCRIPT
manubot ai-revision --content-directory content/ --config-directory ci/
```

The editor will revise each paragraph of your manuscript and write back the revised files in the same directory.
Finally, (_assuming you are tracking changes to your manuscript with git_) you can review each change and either keep it (commit it) or reject it (revert it).

Using the OpenAI API can sometimes incur costs.
If you're worried about this or otherwise want to test things out before hitting the real API, you can run a local "dry run" by with a "fake" model:

```bash
manubot ai-revision \
  --content-directory content/ \
  --config-directory ci/ \
  --model-type DummyManuscriptRevisionModel \
  --model-kwargs add_paragraph_marks=True
```

When it finishes, check out your manuscript files.
This will allow you to detect whether the editor is identifying paragraphs correctly.
If you find a problem, please [report the issue](https://github.com/manubot/manubot-ai-editor/issues).

## Using the Python API

You can also use the functions of the editor directly from Python.

Since these functions are low-level and not tied to a particular manuscript, you don't have to install Manubot and can just install this package:

```bash
pip install -U manubot-ai-editor
```

Example usage:

```python
import shutil
from pathlib import Path

from manubot_ai_editor.editor import ManuscriptEditor
from manubot_ai_editor.models import GPT3CompletionModel

# create a manuscript editor object.
me = ManuscriptEditor(
    # where your Markdown files (*.md) are
    content_dir="content",
    # where CI-related configuration, including the AI editor's, is stored.
    # optional, will fallback to defaults if omitted.
    config_dir="ci"
)

# create a model to revise the manuscript
model = GPT3CompletionModel(
    title=me.title,
    keywords=me.keywords,
)

# create a temporary directory to store the revised manuscript
output_folder = (Path("tmp") / "manubot-ai-editor-output").resolve()
shutil.rmtree(output_folder, ignore_errors=True)
output_folder.mkdir(parents=True, exist_ok=True)

# revise the manuscript
me.revise_manuscript(output_folder, model)

# the revised manuscript is now in the `output_folder`

# uncomment the following code if you want to OVERWRITE the original manuscript in the content folder with the revised manuscript
# for f in output_folder.glob("*"):
#     f.rename(me.content_dir / f.name)
#
# # remove output folder
# output_folder.rmdir()
```

The [`cli_process` function in this file](https://github.com/manubot/manubot/blob/f62dd4cfdebf67f99f63c9b2e64edeaa591eeb69/manubot/ai_revision/ai_revision_command.py#L7) provides another example of how to use the API.
