import re
import difflib

import yaml


SIMPLE_SENTENCE_END_PATTERN = re.compile(r"\.\s")
SENTENCE_END_PATTERN = re.compile(r"\.\s(\S)")


def get_yaml_field(yaml_file, field):
    """
    Returns the value of a field in a YAML file.
    """
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    return data[field]


def starts_with_similar(string: str, prefix: str, threshold: float = 0.8) -> bool:
    """
    Returns True if the string starts with a prefix that is similar to the given prefix.
    """
    return (
        difflib.SequenceMatcher(None, prefix, string[: len(prefix)]).ratio() > threshold
    )
