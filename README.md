# Manubot AI Editor

[![PyPI - Version](https://img.shields.io/pypi/v/manubot-ai-editor)](https://pypi.org/project/manubot-ai-editor/)
[![Build Status](https://github.com/manubot/manubot-ai-editor/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/manubot/manubot-ai-editor/actions/workflows/run-tests.yml?query=branch%3Amain)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Manuscript DOI badge](https://img.shields.io/badge/Manuscript_DOI-10.1093/jamia/ocae139-blue)](https://doi.org/10.1093/jamia/ocae139)
[![Software DOI badge](https://img.shields.io/badge/Software_DOI-10.5281/zenodo.14911573-blue)](https://doi.org/10.5281/zenodo.14911573)

A tool for performing automatic, AI-assisted revisions of [Manubot](https://manubot.org/) manuscripts.
Check out the [manuscript about this tool](https://greenelab.github.io/manubot-gpt-manuscript/) for more background information.

## Supported Large Language Models (LLMs)

We internally use [LangChain](https://www.langchain.com/) to invoke models, which allows our tool to theoretically
support whichever model providers LangChain supports. That said, we currently support OpenAI and Anthropic models only,
and are working to add support for other model providers.

When using OpenAI models, [our evaluations](https://github.com/pivlab/manubot-ai-editor-evals) show that `gpt-4-turbo`
is in general the best model for revising academic manuscripts. Therefore, this is the default option for OpenAI.

We are still evaluating the models for other providers as we add them, and will update this section accordingly
as we complete our evaluations.

## Using in a Manubot manuscript

Much of these instructions rely on the specific details of GitHub's website interface, which can change over time.
See their official docs for more info on [configuring GitHub Actions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository), [managing secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions#creating-secrets-for-a-repository), and [running workflows](https://docs.github.com/en/actions/using-workflows/manually-running-a-workflow).

### Setup

First, you should decide which model provider you'll use. You can find details on how to set up each provider below:
- **OpenAI:** you'll want to [make an OpenAI account](https://openai.com/api/) and [create an API key](https://platform.openai.com/api-keys).
- **Anthropic:** you'll want to [make an Anthropic account](https://www.anthropic.com/api) and [create an API key](https://console.anthropic.com/settings/keys).

Start with a manuscript repo [forked from Manubot rootstock](https://github.com/manubot/rootstock), then follow these steps:

1. In your forks's "▶️ Actions" tab, enable GitHub Actions.
1. In your fork's "⚙️ Settings" tab, give GitHub Actions workflows read/write permissions and allow them to create pull requests.
1. If you haven't already, follow the directions above to create an account and get an API key for your chosen model provider.
1. In your fork's "⚙️ Settings" tab, make a new Actions repository secret with the name `PROVIDER_API_KEY` and paste in your API key as the secret.

If you prefer to select less options when running the workflow, you can optionally set up default values for the model provider and model at either the repo or organization level.

In your fork's "⚙️ Settings" tab, you can optionally create the folllowing Actions repository variables:
- `AI_EDITOR_MODEL_PROVIDER`: Either "openai" or "anthropic"; sets this as the default if "(repo default)" was selected in the workflow parameters.
  If this is unspecified and "(repo default)" is selected, the workflow will throw an error.
- `AI_EDITOR_LANGUAGE_MODEL`: For the given provider, what model to use if the "model" field in the workflow parameters was left empty.
  If this is unspecified, Manubot AI Editor will select the default model for your chosen provider.

### Multiple Providers

In case you want to use several providers in the same repo, you'll have to register an API key for each provider you intend to use.
Like `PROVIDER_API_KEY`, these keys are also registered as GitHub secrets, and can be specified at either the repository or organizational level.

We currently support the following secrets, with more to follow as we integrate more providers:
- `OPENAI_API_KEY`: the API key for the "openai" provider
- `ANTHROPIC_API_KEY`: the API key for the "anthropic" provider

See [the API key variables docs](https://github.com/manubot/manubot-ai-editor/blob/main/docs/env-vars.md#provider-api-key-configuration) for more information.

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

You also need to export an environment variable with your model provider's API key, e.g.:

```bash
export OPENAI_API_KEY=ABCD1234
# export ANTHROPIC_API_KEY=ABCD1234 # if you were using anthropic
```

If you only ever use one model provider (e.g., just OpenAI or just Anthropic), you can alternatively provide just
`PROVIDER_API_KEY` and it will be used with any model provider the tool invokes.

To select a specific provider, set the environment variable `AI_EDITOR_MODEL_PROVIDER` to one of the following values:
- `openai` for OpenAI
- `anthropic` for Anthropic

If `AI_EDITOR_MODEL_PROVIDER` is unset, it will default to "openai".

You can also provide other environment variables that will change the behavior of the editor (such as revising certain files only).
For example, to specify the temperature parameter of OpenAI models, you can set the variable `export AI_EDITOR_TEMPERATURE=0.50`.
See [the complete list of supported variables](https://github.com/manubot/manubot-ai-editor/blob/main/docs/env-vars.md) for
more information.

Then, from the root directory of your Manubot manuscript, run the following:

```bash
# ⚠ THIS WILL OVERWRITE YOUR LOCAL MANUSCRIPT
manubot ai-revision --content-directory content/ --config-directory ci/
```

The editor will revise each paragraph of your manuscript and write back the revised files in the same directory.
Finally, (_assuming you are tracking changes to your manuscript with git_) you can review each change and either keep it (commit it) or reject it (revert it).

Using model providers' APIs can sometimes incur costs.
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

### Text Encodings

By default, Manubot AI Editor will assume that your input and output files are
encoded in the `utf-8` encoding.

If you'd prefer for the tool to make a best effort to guess the input encoding
and write the output in the same encoding, set the env var
`AI_EDITOR_SRC_ENCODING` to `_auto_`; the detected encoding will also be used to
write the output files.

Alternatively, if you prefer to have your files interpreted or written using
specific encodings, you can specify the input encoding with the
`AI_EDITOR_SRC_ENCODING` and the output encoding with the
`AI_EDITOR_DST_ENCODING` environment variables.

See[these variables' help docs](https://github.com/manubot/manubot-ai-editor/blob/main/docs/env-vars.md#encodings)
for more information.

Also, see [Python 3 Docs: Standard Encodings](https://docs.python.org/3/library/codecs.html#standard-encodings) for
a list of possible encodings.

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
# (if using another provider, e.g. anthropic, replace model_provider="openai" with model_provider="anthropic")
model = GPT3CompletionModel(
    title=me.title,
    keywords=me.keywords,
    model_provider="openai",
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

## Development and Contributions

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guide for more information on developing this project or making a contributon.
