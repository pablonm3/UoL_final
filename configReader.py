import toml
config = {}

def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            else:
              a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def get_config(config_file, config_path):
    global config
    toml_path = f"{config_path}/{config_file}.toml"
    if(toml_path in config):
        return config[toml_path]
    default_config = toml.load(f"{config_path}/default.toml")
    print(f"reading default_config: {default_config}")
    try:
        client_config = toml.load(toml_path)
        print(f"read client_config: {default_config}")
    except FileNotFoundError as e:
        config[toml_path] = default_config
        print(f"client config file not available for {config_file}, using final config(default): {config[toml_path]}")
        return config[toml_path]
    config[toml_path] = merge(default_config, client_config)
    return config[toml_path]