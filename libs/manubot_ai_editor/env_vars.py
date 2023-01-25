"""
This file contains environment variables names used by manubot-ai-editor
package. Several of them allow to specify different parameters when calling the
OpenAI model, such as LAGUANGE_MODEL or MAX_TOKENS_PER_REQUEST. For this, see
more details in https://beta.openai.com/docs/api-reference/completions/create
"""

# OpenAI API key to use
OPENAI_API_KEY = "OPENAI_API_KEY"

# Language model to use. For example, "text-davinci-003"
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
