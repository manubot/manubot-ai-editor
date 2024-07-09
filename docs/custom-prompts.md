# Custom Prompts

Rather than using the default prompt, you can specify custom prompts for each file in your manuscript.
This can be useful when you want specific sections of your manuscript to be revised in specific ways, or not revised at all.

There are two ways that you can use the custom prompts system:
1. You can define your prompts and how they map to your manuscript files in a single file, `ai_revision-prompts.yaml`.
2. You can create the `ai_revision-prompts.yaml`, but only specify prompts and identifiers, which makes it suitable for sharing with others who have different names for their manuscripts' files.
You would then specify a second file, `ai_revision-config.yaml`, that maps the prompt identifiers to the actual files in your manuscript.

These files should be placed in the `content` directory alongside your manuscript markdown files.

See [Example Configuration](#example-configuration) for a quick guide on how to enable the custom prompts system.

See [Functionality Notes](#functionality-notes) later in this document for more information on how to write regular expressions and use placeholders in your prompts.


## Approach 1: Single file

With this approach, you can define your prompts and how they map to your manuscript files in a single file.
The single file should be named `ai_revision-prompts.yaml` and placed in the `content` folder.

The file would look something like the following:

```yaml
prompts_files:
  # filenames are specified as regular expressions
  # in this case, we match a file named exactly 'filename.md'
  ^filename\.md$: "Prompt text here"

  # you can use YAML's multi-line string syntax to write longer prompts
  # you can also use {placeholders} to include metadata from your manuscript
  ^filename\.md$: |
    Revise the following paragraph from a manuscript titled {title}
    so that it sounds like an academic paper.

  # specifying the special value 'null' will skip revising any files that
  # match this regular expression
  ^ignore_this_file\.md$: null
```

Note that, for each file, the first matching regular expression will determine its prompt or whether the file is skipped.
Even if a file matches multiple regexes, only the first one will be used.


## Approach 2: Prompt file plus configuration file

In this case, we specify two files, `ai_revision-prompts.yaml` and `ai_revision-config.yaml`.

The `ai_revision-prompts.yaml` file contains only the prompts and their identifiers.
The top-level element is `prompts` in this case rather than `prompts_files`, as it defines a set of resuable prompts and not prompt-file mappings.

Here's an example of what the `ai_revision-prompts.yaml` file might look like:
```yaml
prompts:
  intro_prompt: "Prompt text here"
  content_prompts: |
    Revise the following paragraph from a manuscript titled {title}
    so that it sounds like an academic paper.

  my_default: "Revise this paragraph so it sounds nicer."
```

The `ai_revision-config.yaml` file maps the prompt identifiers to the actual files in your manuscript.

An example of the `ai_revision-config.yaml` file:
```yaml
files:
  matchings:
    - files:
        - ^introduction\.md$
      prompt: intro_prompt
    - files:
        - ^methods\.md$
        - ^abstract\.md$
      prompt: content_prompts

  # the special value default_prompt is used when no other regex matches
  # it also uses a prompt identifier taken from ai_revision-prompts.yaml
  default_prompt: my_default

  # any file you want to be skipped can be specified in this list
  ignores:
    - ^ignore_this_file\.md$
```

Multiple regexes can be specified in a list under `files` to match multiple files to a single prompt.

In this case, the `default_prompt` is used when no other regex matches, and it uses a prompt identifier taken from `ai_revision-prompts.yaml`.

The `ignores` list specifies files that should be skipped entirely during the revision process; they won't have the default prompt applied to them.

## Example Configuration

You can find an example of the `ai_revision-prompts.yaml` and `ai_revision-config.yaml` files in the `docs/example` directory of this repository.

The prompts file, [`docs/example/ai_revision-prompts.yaml`](docs/example/ai_revision-prompts.yaml), contains some example prompts as well as a default prompt.
The config file, [`docs/example/ai_revision-config.yaml`](docs/example/ai_revision-config.yaml), maps all `.md` files to the prompt with the identifier `default`.

You can copy these two files into your manuscript's `content` directory to start using the custom prompts system, then modify them to suit your needs.

## Functionality Notes

### Filenames as Regular Expressions

Filenames in either approach are specified as regular expressions (aka "regexes").
This allows you to flexibly match multiple files to a prompt with a single expression.

A simple example: to specify an exact match for, say, `myfile.md`, you'd supply the regular expression `^myfile\.md$`, where:
- `^` matches the beginning of the filename
- `\.` matches a literal period -- otherwise, `.` means "any character"
- `$` matches the end of the filename

To illustrate why that syntax is important: if you were to write it as `myfile.md`, the `.` would match any character, so it would match `myfileAmd`, `myfile2md`, etc.
Without the `^` and `$`, it would match also match filenames like `asdf_myfile.md`, `myfile.md_asdf`, and `asdf_myfile.md.txt`.

The benefit of using regexes becomes more apparent when you have multiple files.
For example, say you had three files, `02.prior-work.md`, `02.methods.md`, and `02.results.md`. To match all of these, you could use the expression `^02\..*\.md$`.
This would match any file beginning with `02.` and ending with `.md`.
Here, `.` again indicates "any character" and the `*` means "zero or more of the preceding character; together, they match any sequence of characters.

You can find more information on how to write regular expressions in [Python's `re` module documentation](https://docs.python.org/3/library/re.html#regular-expression-syntax).


### Placeholders

The prompt text can include metadata from your manuscript, specified in `content/metadata.yaml` in Manubot. Writing
`{placeholder}` into your prompt text will cause it to be replaced with the corresponding value, drawn either
from the manuscript metadata or from the current file/paragraph being revised.

The following placeholders are available:
- `{title}`: the title of the manuscript, as defined in the metadata
- `{keywords}`: comma-delimited keywords from the manuscript metadata
- `{paragraph_text}`: the text from the current paragraph
- `{section_name}`: the name of the section (which is one of the following values "abstract",  "introduction", "results", "discussion", "conclusions", "methods" or "supplementary material"), derived from the filename.

The `section_name` placeholder works like so:
- if the env var `AI_EDITOR_FILENAME_SECTION_MAPPING` is specified, it will be interpreted as a dictionary mapping filenames to section names.
If a key of the dictionary is included in the filename, the value will be used as the section name.
Also the keys and values can be any string, not just one of the section names mentioned before.
- If the dict mentioned above is unset or the filename doesn't match any of its keys, the filename will be matched against the following values: "introduction", "methods", "results", "discussion", "conclusions" or "supplementary".
If the values are contained within the filename, the section name will be mapped to that value. "supplementary" is replaced with "supplementary material", but the others are used as is.
