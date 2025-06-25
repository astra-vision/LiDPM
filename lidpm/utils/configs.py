import os.path

import yaml
import click


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def save_config(config, file_path):
    os.makedirs(file_path.rsplit('/', 1)[0], exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)


def update_config(config, args):
    for key, value in vars(args).items():
        if value is not None and key != 'config_path' and key != 'save':
            config[key] = value


def smart_config(config_path, ctx):
    """
    Load config, update it with the command-line options, print it.
    :param config_path: path to the .yaml config file
    :param ctx: click.get_current_context() output
    :return: updated config
    """
    config = load_config(config_path)

    options = ctx.params

    for key, value in options.items():
        if value is not None:
            config[key] = options[key]

    click.echo(yaml.safe_dump(config))

    return config
