import yaml


class DotConfig:
    """
    Class for making configuration file accessible with dot.
    """
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v


def read_config(config_file_path):
    """
    Read configuration file.
    Args:
        config_file_path: Path to configuration file.

    Returns: Configuration parameters as dictionary.

    """
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return DotConfig(config)

