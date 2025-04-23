# Contributing

First of all, thank you so much for contributing! 🎉 💯

This document contains guidelines on how to most effectively contribute within this repository.

If you are stuck, please feel free to ask any questions or ask for help.

## Code of conduct

This project is governed by our [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Development

This project leverages development environments managed by a Python [Poetry `pyproject.toml` file](https://python-poetry.org/docs/).
We use [pytest](https://docs.pytest.org/) for testing and [GitHub Actions](https://docs.github.com/en/actions) for automated tests.
[`pre-commit`](https://pre-commit.com/) is used to help lint or format code.
A [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is provided (via the file `environment.yml`) for convenience but is not required for development purposes.

### Development setup

Perform the following steps to setup a Python development environment.

1. [Install Python](https://www.python.org/downloads/) (we recommend using [`pyenv`](https://github.com/pyenv/pyenv) or similar)
1. [Install Poetry](https://python-poetry.org/docs/#installation).
1. Install package, e.g. `poetry install`

### Linting

Work added to this project is automatically checked using [pre-commit](https://pre-commit.com/) via [GitHub Actions](https://docs.github.com/en/actions).
Pre-commit can work alongside your local [git with git-hooks](https://pre-commit.com/index.html#3-install-the-git-hook-scripts)

After [installing pre-commit](https://pre-commit.com/#installation) within your development environment, the following command also can perform the same checks within your local development environment:

```sh
% pre-commit run --all-files
```

We use these same checks within our automated tests which are managed by [GitHub Actions workflows](https://docs.github.com/en/actions/using-workflows).
These automated tests generally must pass in order to merge work into this repository.

### Testing

Contributions to this project are automatically tested using [pytest](https://docs.pytest.org/) via [GitHub Actions](https://docs.github.com/en/actions).
Pytest is installed through the Poetry environment for this project.
We recommend adding tests for your contributions, if they're not already covered by an existing test.
Once you've written your tests, we encourage running the testing suite before opening a pull request for your contribution.

Many tests require a special environment key to be set: `PROVIDER_API_KEY`.
While the `PROVIDER_API_KEY` environment variable must be set to a value for the testing suite to complete, any non-whitespace value will do.
It doesn't have to be a valid API key, since all tests that actually query the API with the key are skipped by default.

Once `PROVIDER_API_KEY` is set to something, you can run `pytest` on your work using the following example:
```sh
% poetry run pytest
```

If you want to run tests that query the actual provider APIs, you must specify a valid API key for each provider that's tested.
If you're testing just one provider, it's sufficient to specify `PROVIDER_API_KEY`, since by default it will be used for any provider.

For reference, API keys are resolved in the following order:
1. The `api_key` argument to the `models.GPT3CompletionModel` constructor will always be used first.
2. API-specific keys specified via environment variables, e.g. `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, will then be
   used. The specific key used depends on which model provider (e.g., `openai`, `anthropic`) is in use.
3. If no API-specific key is found, the environment variable `PROVIDER_API_KEY` will be used.

We currently support the following providers:
- **OpenAI:** Set `OPENAI_API_KEY` to a valid key.
  - See here to [make an OpenAI account](https://openai.com/api/) and [create an API key](https://platform.openai.com/api-keys).
- **Anthropic:** Set `ANTHROPIC_API_KEY` to a valid key.
  - See here to [make an Anthropic account](https://console.anthropic.com/) and [create an API key](https://console.anthropic.com/settings/keys).

You can set these keys like so:

```bash
export OPENAI_API_KEY=ABCD1234
export ANTHROPIC_API_KEY=ABCD1234
```

You can then execute `pytest` with the `--runcost` option, e.g. `poetry run pytest --runcost`, which will then use the specified key to run tests that query the API.
Note that running with `--runcost` will cost you money, typically around a cent or two per execution, depending on your choice of model.
For OpenAI, you can find detailed information about cost for each model per 1M tokens on the [OpenAI API Pricing Page](https://openai.com/api/pricing/) -- our test suite uses `gpt-3.5-turbo`.
At the moment there is no way to only test a subset of providers.
Instead, you can provide made-up keys for providers you don't want to actually test and ignore the resulting failures.

The "cost" tests take significantly longer than the non-live test suite to complete, so it's recommended to leave them disabled them unless you know you need to test the live APIs.


## Making changes to this repository

We welcome anyone to use [GitHub issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues) (requires a GitHub login) or create [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) (to directly make changes within this repository) to modify content found within this repository.

Specifically, there are several ways to suggest or make changes to this repository:

1. Open a GitHub issue: https://github.com/manubot/manubot-ai-editor/issues
1. Create a pull request from a forked branch of the repository

### Pull requests

After you’ve decided to contribute code and have written it up, please file a pull request.
We specifically follow a [forked pull request model](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).
Please create a fork of this repository, clone the fork, and then create a new, feature-specific branch.
Once you make the necessary changes on this branch, you should file a pull request to incorporate your changes into this (fork upstream) repository.

The content and description of your pull request are directly related to the speed at which we are able to review, approve, and merge your contribution.
To ensure an efficient review process please perform the following steps:

1. Triple check that your pull request is adding _one_ specific feature or additional group of content.
   Small, bite-sized pull requests move so much faster than large pull requests.
1. After submitting your pull request, ensure that your contribution passes all status checks (e.g. passes all tests)

Pull request review and approval is required by at least one project maintainer to merge.
We will do our best to review the code addition in a timely fashion.
Ensuring that you follow all steps above will increase our speed and ability to review.
We will check for accuracy, style, code coverage, and scope.

## Versioning

We use [`poetry-dynamic-versioning`](https://github.com/mtkennerly/poetry-dynamic-versioning) to help version this software through [`PEP 440`](https://peps.python.org/pep-0440/) standards.
Configuration for versioning is found within the `pyproject.toml` file.
All builds for packages include dynamic version data to help label distinct versions of the software.
`poetry-dynamic-versioning` uses `git` tags to help distinguish version data.
We also use the `__init__.py` file as a place to persist the version data for occaissions where the `git` history is unavailable or unwanted.

The following command is used to add `poetry-dynamic-versioning` to Poetry for use with this project: `poetry self add "poetry-dynamic-versioning[plugin]"`.
Versioning for the project is intended to align with GitHub Releases which provide `git` tag capabilities.

### Releases

We publish source code by using [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) available [here](https://github.com/manubot/manubot-ai-editor/releases).
We publish a related Python package through the [Python Packaging Index (PyPI)](https://pypi.org/) available [here](https://pypi.org/project/manubot-ai-editor/).

#### Release Publishing Process

Several manual and automated steps are involved with publishing manubot-ai-editor releases.
See below for an overview of how this works.

Notes about [semantic version](https://en.wikipedia.org/wiki/Software_versioning#Semantic_versioning) (semver) specifications:
manubot-ai-editor version specifications are controlled through [`poetry-dynamic-versioning`](https://github.com/mtkennerly/poetry-dynamic-versioning) which leverages [`dunamai`](https://github.com/mtkennerly/dunamai) to create version data based on [git tags](https://git-scm.com/book/en/v2/Git-Basics-Tagging) and commits.
manubot-ai-editor release git tags are automatically applied through [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) and related inferred changes from [`release-drafter`](https://github.com/release-drafter/release-drafter).

1. Open a pull request and use a repository label for `release-<semver release type>` to label the pull request for visibility with [`release-drafter`](https://github.com/release-drafter/release-drafter).
1. On merging the pull request for the release, a [GitHub Actions workflow](https://docs.github.com/en/actions/using-workflows) defined in `draft-release.yml` leveraging [`release-drafter`](https://github.com/release-drafter/release-drafter) will draft a release for maintainers.
1. The draft GitHub release will include a version tag based on the GitHub PR label applied and `release-drafter`.
1. Make modifications as necessary to the draft GitHub release, then publish the release (the draft release does not require additional modifications by default).
1. On publishing the GitHub release, another GitHub Actions workflow defined in `publish-pypi.yml` will run to build and deploy the Python package to PyPI (utilizing the earlier modified `pyproject.toml` semantic version reference for labeling the release).