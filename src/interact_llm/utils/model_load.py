from pathlib import Path
from typing import Literal
import toml

def get_model_id(models_config_path: Path, model_name: str, backend: Literal["mlx", "hf"] = "mlx") -> str:
    """
    Reads models from a TOML file and returns the correct model ID 
    based on the specified backend ('mlx' or 'hf') and an input model ID.
    
    models_config_path: Path to the models.toml file (usually placed in /configs)
    model_name: name defined in toml with corresponding backends
    backend: Either 'mlx' or 'hf'

    returns: 
        The corresponding model ID for the selected backend
    """
    if models_config_path.suffix != '.toml':
        raise ValueError(f"The file at '{models_config_path}' is not a TOML file.")

    if backend not in {"mlx", "hf"}:
        raise ValueError("Backend must be 'mlx' or 'hf'")

    models = toml.load(models_config_path)["models"]

    for model in models:
        if model_name in model.values():
            if backend in model:
                return model[backend]
            raise ValueError(f"Model '{model_name}' exists but has no backend '{backend}' entry.")

    model_names = [model["name"] for model in models]
    raise ValueError(f"No model defined for '{model_name}. Add it to '{models_config_path.name}' or choose between defined models: {model_names}'")

def login_hf_token(token_path=Path(__file__).parents[3] / "tokens" / "hf_token.txt"): 
    '''
    Load HF token from "tokens" folder and login. 
    '''
    from huggingface_hub import login

    # get token from txt
    with open(token_path) as f:
        hf_token = f.read()

    login(hf_token)
