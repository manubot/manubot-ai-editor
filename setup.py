import setuptools

# Commands to publish new package:
#
# rm -rf dist/
# python setup.py sdist
# twine upload dist/*

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="manubot-chatgpt-editor",
    version="0.0.1",
    author="Milton Pividori",
    author_email="miltondp@gmail.com",
    description="A Manubot plugin to revise a manuscript using GPT-3",
    license="BSD-2-Clause Plus Patent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miltondp/manubot-chatgpt-editor",
    package_dir={"": "libs"},
    packages=[
        "chatgpt_editor/",
    ],
    python_requires=">=3.10",
    install_requires=[
        "openai>=0.25",
        "pyyaml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
