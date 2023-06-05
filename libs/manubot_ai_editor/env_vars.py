"""
This file contains environment variables names used by manubot-ai-editor
package. They allow to specify different parameters when calling the
OpenAI model, such as the language model or the maximum tokens per request
(see more details in https://beta.openai.com/docs/api-reference/completions/create).

If you are using our GitHub Actions workflow provided by manubot/rootstock, you need
to modify the "Revise manuscript" step in the workflow file (.github/workflows/ai-revision.yaml)
by adding the environment variable name specificed in the _value_ of the variables. For instance,
if you want to provide a custom prompt, then you need to add a line like this to the workflow:

    AI_EDITOR_CUSTOM_PROMPT="proofread the following paragraph"
"""

# OpenAI API key to use
OPENAI_API_KEY = "OPENAI_API_KEY"

# Language model to use. For example, "text-davinci-003", "gpt-3.5-turbo", "gpt-3.5-turbo-0301", etc
# The tool currently supports the "chat/completions", "completions", and "edits" endpoints, and you can check
# compatible models here: https://platform.openai.com/docs/models/model-endpoint-compatibility
LANGUAGE_MODEL = "AI_EDITOR_LANGUAGE_MODEL"

# Model parameter: max_tokens
MAX_TOKENS_PER_REQUEST = "AI_EDITOR_MAX_TOKENS_PER_REQUEST"

# Model parameter: temperature
TEMPERATURE = "AI_EDITOR_TEMPERATURE"

# Model parameter: top_p
TOP_P = "AI_EDITOR_TOP_P"

# Model parameter: presence_penalty
PRESENCE_PENALTY = "AI_EDITOR_PRESENCE_PENALTY"

# Model parameter: frequency_penalty
FREQUENCY_PENALTY = "AI_EDITOR_FREQUENCY_PENALTY"

# Model parameter: best_of
BEST_OF = "AI_EDITOR_BEST_OF"

# It allows to specify a JSON string, where keys are filenames and values are
# section names. For example: '{"01.intro.md": "introduction"}'
# Possible values for section names are: "abstract", "introduction",
# "results", "discussion", "conclusions", "methods", and "supplementary material".
# Take a look at function 'get_prompt' in 'libs/manubot_ai_editor/models.py'
# to see which prompts are used for each section.
# Although the AI Editor tries to infer the section name from the filename,
# sometimes filenames are not descriptive enough (e.g., "01.intro.md" or
# "02.review.md" might indicate an introduction).
# Mapping filenames to section names is useful to provide more context to the
# AI model when revising a paragraph. For example, for the introduction, prompts
# contain sentences to preserve most of the citations to other papers.
SECTIONS_MAPPING = "AI_EDITOR_FILENAME_SECTION_MAPPING"

# Sometimes the AI model returns an empty paragraph. Usually, this is resolved
# by running again the model. The AI Editor will try five (5) times in these
# cases. This variable allows to specify the number of retries.
RETRY_COUNT = "AI_EDITOR_RETRY_COUNT"

# If specified, only these file names will be revised. Multiple files can be
# specified, separated by commas. For example: "01.intro.md,02.review.md"
FILENAMES_TO_REVISE = "AI_EDITOR_FILENAMES_TO_REVISE"

# It allows to specify a single, custom prompt for all sections. For example:
# "proofread and revise the following paragraph"; in this case, the tool will automatically
# append the characters ':\n\n' followed by the paragraph.
# It is also possible to include placeholders in the prompt, which will be replaced
# by the corresponding values. For example, "proofread and revise the following
# paragraph from the section {section_name} of a scientific manuscript with title '{title}'".
# The complete list of placeholders is: {paragraph_text}, {section_name},
# {title}, {keywords}.
CUSTOM_PROMPT = "AI_EDITOR_CUSTOM_PROMPT"
