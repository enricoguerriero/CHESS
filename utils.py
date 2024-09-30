import yaml

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model = config['model'] 
    data = config['data']
    pretraining = config['pretraining']
    training = config['training']
    return model, data, pretraining, training   

def load_model(model_path: str) -> dict:
    """Load a model from a file."""
    with open(model_path, 'r') as file:
        model = yaml.safe_load(file)
    return model
