import importlib  # Import the importlib module to dynamically import modules
import os.path as osp  # Import os.path module for path operations and alias it as osp

def get_config(config_file):
    """
    Dynamically loads and merges configuration settings from a specified config file.

    Args:
        config_file (str): The path to the config file, which must start with 'configs/'.

    Returns:
        cfg (dict): The merged configuration dictionary.
    """
    # Ensure the config file path starts with 'configs/'
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'

    # Extract the base name of the config file (e.g., 'example_config.py')
    temp_config_name = osp.basename(config_file)
    
    # Remove the file extension to get the module name (e.g., 'example_config')
    temp_module_name = osp.splitext(temp_config_name)[0]
    
    # Import the base configuration module
    config = importlib.import_module("configs.base")
    
    # Get the base configuration dictionary
    cfg = config.config
    
    # Import the specific job configuration module
    config = importlib.import_module("configs.%s" % temp_module_name)
    
    # Get the job-specific configuration dictionary
    job_cfg = config.config
    
    # Merge the job-specific configuration into the base configuration
    cfg.update(job_cfg)
    
    # Set the result directory if not already specified
    if cfg.result is None:
        cfg.result = osp.join('work_dirs', temp_module_name)
    
    # Return the merged configuration
    return cfg