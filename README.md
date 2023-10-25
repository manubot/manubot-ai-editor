# Manubot AI Editor

This package provides classes and functions for automated, AI-assisted revision of manuscripts written using [Manubot](https://manubot.org/).
Check out the [manuscript](https://greenelab.github.io/manubot-gpt-manuscript/) for more information.

## Usage

The Manubot AI Editor can be used from the GitHub repository of a Manubot-based manuscript, from the command line, or from Python code.

### Manubot-based manuscript repository

You first need to follow the steps to [setup a Manubot-based manuscript](https://github.com/manubot/rootstock).
Then, follow [these instructions](https://github.com/manubot/rootstock/blob/main/USAGE.md#ai-assisted-authoring) to setup a workflow in GitHub Actions that will allow you to quickly trigger a job to revise your manuscript.

### Command line

To use the tool from the command line, you first need to install Manubot in a Python environment:

```bash
pip install --upgrade manubot[ai-rev]
```

You also need to export an environment variable with the OpenAI's API key:

```bash
export OPENAI_API_KEY=<your-api-key>
```

You can also provide other options that will change the behavior of the tool (such as revising certain files only).
[This file](https://github.com/manubot/manubot-ai-editor/blob/main/libs/manubot_ai_editor/env_vars.py) documents the list of supported environment variables that can be used.
For example, to change the temperature parameter of OpenAI models, you can export the following environment variable: `export AI_EDITOR_TEMPERATURE=0.50`

Then, within the root directory of your Manubot-based manuscript, run the following commands (**IMPORTANT:** this will overwrite your original manuscript!):

```bash
manubot ai-revision --content-directory content/
```

The tool will revise each paragraph of your manuscript and write back the revised files in the same directory.
Finally, you can select which changes you want to keep or discard.

### Python API

There is also a Python API that you can use to revise your manuscript.
In this case, you don't need to also install Manubot but only this package:

```bash
pip install -U manubot-ai-editor
```

The Python code below shows how to use the API:

```python
import shutil
from pathlib import Path

from manubot_ai_editor.editor import ManuscriptEditor
from manubot_ai_editor.models import GPT3CompletionModel

# create a manuscript editor object
# here content_dir points to the "content" directory of the Manubot-based
# manuscript, where Markdown files (*.md) are located
me = ManuscriptEditor(
    content_dir="content",
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

# then revise the manuscript
me.revise_manuscript(output_folder, model)

# the revised manuscript is now in the folder pointed by `output_folder`

# uncomment the following code if you want to write back the revised manuscript to
# the content folder
# **CAUTION**: this will overwrite the original manuscript
# for f in output_folder.glob("*"):
#     f.rename(me.content_dir / f.name)
# 
# # remove output folder
# output_folder.rmdir()
```

The `cli_process` function in [this file](https://github.com/manubot/manubot/blob/f62dd4cfdebf67f99f63c9b2e64edeaa591eeb69/manubot/ai_revision/ai_revision_command.py#L7) also provides an example of how to use the API.

## Current support for large language models

We currently support the following OpenAI endpoints:
* [`Completion`](https://platform.openai.com/docs/api-reference/completions)
* [`Edits`](https://platform.openai.com/docs/api-reference/edits)
* [`ChatCompletion`](https://platform.openai.com/docs/api-reference/chat)
  * *Note:* this endpoint is not fully implemented yet.
    The current implementation uses the chat completion endpoint in a similar way as we use the completion endpoint (each paragraph is revised independently in a query).
    This is because new models such as `gpt-3.5-turbo` or `gpt-4` are only available through the chat completion endpoint.
