import os.path

import yaml
import click


# Function to load the YAML configuration file
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


# Function to save the YAML configuration file
def save_config(config, file_path):
    # check if the folder exists
    os.makedirs(file_path.rsplit('/', 1)[0], exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)


# Function to update the configuration with command-line arguments
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
    # Load the default configuration
    config = load_config(config_path)

    # Get the command-line options as a dictionary
    options = ctx.params

    # Update the configuration with command-line arguments
    for key, value in options.items():
        if value is not None:
            config[key] = options[key]

    # Print the final configuration
    click.echo(yaml.safe_dump(config))

    return config
