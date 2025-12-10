import os
import yaml
from collections.abc import Mapping


def load(key=None, file='default.yaml'):
    """
    Import default configuration from config.yaml or user-specific configuration
    from user.yaml.
    Args:
      key = can specify the highest level key, e.g. 'area', if only a specific
        configuration information is required.
      file = str name of the config file.
    Returns:
      cgf = a dictionary with all relevant configuration parameters.
    """

    path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(path, file)
    fh = open(file_path)
    cfg = yaml.load(fh, Loader=yaml.FullLoader)
    fh.close()

    try:
        fh = open(os.path.join(path, 'user.yaml'))
        cfg = recursive_dict_update(cfg, yaml.load(fh, Loader=yaml.FullLoader))
    except IOError:
        print('\'user.yaml\' not found. Using default configuration.')

    if key is None:
        return cfg
    else:
        return cfg[key]

    

def recursive_dict_update(d, u):
    """Recursive dictionary update
        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """  

    for k, v in u.items():
        if isinstance(v, Mapping):
            r = recursive_dict_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d



if __name__ == '__main__':
    cfg = Config()
