import re

import yaml


SENTENCE_END_PATTERN = re.compile(r"\.\s(\S)")


def get_yaml_field(yaml_file, field):
    """
    Returns the value of a field in a YAML file.
    """
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    return data[field]
