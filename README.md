# Manubot AI Editor

This package provides classes and functions for automated, AI-assisted revision of manuscripts written using [Manubot](https://manubot.org/).
Check out the [manuscript](https://github.com/greenelab/manubot-gpt-manuscript) for more information.

## Support for large language models

We currently support the following OpenAI endpoints:
* [`Completion`](https://platform.openai.com/docs/api-reference/completions)
* [`Edits`](https://platform.openai.com/docs/api-reference/edits)
* [`ChatCompletion`](https://platform.openai.com/docs/api-reference/chat)
  * *Note:* this endpoint is not fully implemented yet.
    The current implementation uses the chat completion endpoint in a similar way as we use the completion endpoint (each paragraph is revised independently in a query).
    This is because new models such as `gpt-3.5-turbo` or `gpt-4` are only available through the chat completion endpoint. 

## Installation

```bash
pip install -U manubot-ai-editor
```

## Usage

The Manubot AI Editor can be used from the GitHub repository of a Manubot-based manuscript, from the command line, or from Python code.

### Manubot-based manuscript GitHub repository

You first need to follow the steps to setup a Manaubot-based manuscript.
Then, follow [these instructions](https://github.com/manubot/rootstock/blob/main/USAGE.md#ai-assisted-authoring) to setup a workflow in GitHub Actions that will allow you to quickly trigger a job to revise your manuscript.

### Command line

To use the tool from the command line, you need to install Manubot in a Python environment:

```bash
pip install --upgrade manubot[ai-rev]
```

You also need to export an environment variable with the OpenAI's API key:

```bash
export OPENAI_API_KEY=<your-api-key>
```

You can also provide other options that will change the behavior of the tool (such as revising certain files only).
[This file](https://github.com/manubot/manubot-ai-editor/blob/main/libs/manubot_ai_editor/env_vars.py) documents the list of supported environment variables that can be used.

Then, within the root directory of your Manubot-based manuscript, run the following commands (**IMPORTANT:** this will overwrite your original manuscript!):

```bash

```bash
manubot ai-revision --content-directory content/
```

The tool will revise each paragraph of your manuscript and write back the revised files in the same directory.

### Python API

There is also a Python API that you can easily use to revise your manuscript.
Take a look at the `cli_process` function in [this file](https://github.com/manubot/manubot/blob/f62dd4cfdebf67f99f63c9b2e64edeaa591eeb69/manubot/ai_revision/ai_revision_command.py#L7) to see how to use it.
You can also take a look at the [unit tests](tests/).
