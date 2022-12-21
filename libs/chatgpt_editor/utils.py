import yaml


def get_yaml_field(yaml_file, field):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data[field]
