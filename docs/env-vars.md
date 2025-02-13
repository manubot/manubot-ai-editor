# Manubot AI Editor Environment Variables

Manubot AI Editor provides a variety of options to customize the revision
process. These options are exposed as environment variables, all of which are
prefixed with `AI_EDITOR_`.

The following environment variables are supported, organized into categories:

## Model Configuration

- `AI_EDITOR_LANGUAGE_MODEL`: Language model to use. For example,
"text-davinci-003", "gpt-3.5-turbo", "gpt-3.5-turbo-0301", etc. The tool
currently supports the "chat/completions", "completions", and "edits" endpoints,
and you can check compatible models here:
https://platform.openai.com/docs/models/model-endpoint-compatibility
- `AI_EDITOR_MAX_TOKENS_PER_REQUEST`: Model parameter: `max_tokens`
- `AI_EDITOR_TEMPERATURE`: Model parameter: `temperature`
- `AI_EDITOR_TOP_P`: Model parameter: `top_p`
- `AI_EDITOR_PRESENCE_PENALTY`: Model parameter: `presence_penalty`
- `AI_EDITOR_FREQUENCY_PENALTY`: Model parameter: `frequency_penalty`
- `AI_EDITOR_BEST_OF`: Model parameter: `best_of`

## Prompt and Query Control

- `AI_EDITOR_FILENAME_SECTION_MAPPING`: Allows the user to specify a JSON
string, where keys are filenames and values are section names. For example:
`{"01.intro.md": "introduction"}` Possible values for section names are:
"abstract", "introduction", "results", "discussion", "conclusions", "methods",
and "supplementary material". Take a look at function `get_prompt()` in
[libs/manubot_ai_editor/models.py](https://github.com/manubot/manubot-ai-editor/blob/main/libs/manubot_ai_editor/models.py#L256)
to see which prompts are used for each section. Although the AI Editor tries to
infer the section name from the filename, sometimes filenames are not
descriptive enough (e.g., "01.intro.md" or "02.review.md" might indicate an
introduction). Mapping filenames to section names is useful to provide more
context to the AI model when revising a paragraph. For example, for the
introduction, prompts contain sentences to preserve most of the citations to
other papers.
- `AI_EDITOR_RETRY_COUNT`: Sometimes the AI model returns an empty paragraph.
Usually, this is resolved by running again the model. The AI Editor will try
five times in these cases. This variable allows to override the number of
retries from its default of 5.
- `AI_EDITOR_FILENAMES_TO_REVISE`: If specified, only these file names will be
revised. Multiple files can be specified, separated by commas. For example:
"01.intro.md,02.review.md"
- `AI_EDITOR_CUSTOM_PROMPT`: Allows the user to specify a single, custom prompt
for all sections. For example: "proofread and revise the following paragraph";
in this case, the tool will automatically append the characters ':\n\n' followed
by the paragraph. It is also possible to include placeholders in the prompt,
which will be replaced by the corresponding values. For example, "proofread and
revise the following paragraph from the section {section_name} of a scientific
manuscript with title '{title}'". The complete list of placeholders is:
`{paragraph_text}`, `{section_name}`, `{title}`, `{keywords}`.

## Encodings

These vars specify the source and destination encodings of input and output markdown
files. Behavior is as follows:
- If neither `SRC_ENCODING` nor `DEST_ENCODING` are specified, both the input
  and output encodings will default to `utf-8`.
- If only `SRC_ENCODING` is specified, it will be used to both read and write
  the files. If the special value `_auto_` is used, the tool will attempt to
  identify the encoding using the
  [charset_normalizer](https://github.com/jawah/charset_normalizer) library,
  then use that encoding to both read the input files and write the output
  files.
- If only `DEST_ENCODING` is specified, it will be used to write the output
  files; the input encoding will be assumed to be `utf-8`.

The variables:

- `AI_EDITOR_SRC_ENCODING`: the encoding of the input markdown files
  - if empty, defaults to `utf-8`, and
  - if `_auto_`, the input encoding is auto-detected.
- `AI_EDITOR_DEST_ENCODING`: the encoding to use when writing the output markdown
  files
  - if empty, defaults to whatever was used for the source encoding.
