# Manubot AI Editor Release Process

This document outlines the process for releasing a new version of the Manubot AI Editor.

*Note that this is a draft and may be subject to change.*

## 1. Update setup.py

Update the line `version="XXX"` in `setup.py` to the new version number. Replace
`XXX` with the new version number, e.g. `0.5.2`.

Make and push a commit including the bumped version number like so:

```bash
git add setup.py
git commit -m "setup.py: update version to XXX"
git push
```

## 2. Create a Git tag for the new version

Create a Git tag for the new version, e.g.:

```bash
git tag -a vXXX -m "vXXX"
git push origin vXXX
```

## 3. Build the package

```bash
rm -rf dist/
python setup.py sdist bdist_wheel
```

## 4. Upload the package to PyPI

Note that you'll require at least "Maintainer" status for the package on PyPI.
If you need access, ask an admin to add you to the list:
https://pypi.org/manage/project/manubot-ai-editor/collaboration/.

During the upload process you'll be prompted for your API token, which you can
generate on https://pypi.org/manage/project/manubot-ai-editor/settings/. You'll
only be able to see it once, so make sure to store it securely.

```bash
twine upload dist/*
```

## 5. Create a release on GitHub

You can visit https://github.com/manubot/manubot-ai-editor/releases/new
to draft a new release. You'll be able to select the tag that you
pushed in step 2.

The "Generate Release Notes" button is pretty handy: it will automatically
populate the release notes based on the commits since the last release
along with other notable events, e.g. new contributors.
