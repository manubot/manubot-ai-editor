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


def get_obj_path(target: any, path: tuple, missing=None):
    """
    Traverse a nested object using a tuple of keys, returning the last resolved
    value in the path. If any key is not found, return 'missing' (default None).

    >>> get_obj_path({'a': {'b': {'c': 1}}}, ('a', 'b', 'c'))
    1
    >>> get_obj_path({'a': {'b': {'c': 1}}}, ('a', 'b', 'd')) is None
    True
    >>> get_obj_path({'a': {'b': {'c': 1}}}, ('a', 'b', 'd'), missing=2)
    2
    >>> get_obj_path({'a': [100, {'c': 1}]}, ('a', 1, 'c'))
    1
    >>> get_obj_path({'a': [100, {'c': 1}]}, ('a', 1, 'd')) is None
    True
    >>> get_obj_path({'a': [100, {'c': 1}]}, ('a', 3)) is None
    True
    """
    try:
        for key in path:
            target = target[key]
    except (KeyError, IndexError, TypeError):
        return missing

    return target
