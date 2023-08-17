import yaml


def read_yaml(path):
    """
    
    """
    content = []
    with open(path, "r") as stream:
        content = yaml.safe_load(stream)

    return content