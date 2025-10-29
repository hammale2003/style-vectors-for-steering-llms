import os
import transformers
from utils.steering_layer import SteeringLayer
from dotenv import load_dotenv

# load environment variables
load_dotenv()

def _resolve_model_id_or_path():
    """Resolve model identifier or local path from environment.

    Priority:
    1) MODEL_ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3" or local folder)
    2) ALPACA_WEIGHTS_FOLDER (legacy var from original repo)
    """
    model_id = os.getenv("MODEL_ID")
    if model_id and len(model_id.strip()) > 0:
        return model_id
    legacy = os.getenv("ALPACA_WEIGHTS_FOLDER")
    if legacy and len(legacy.strip()) > 0:
        return legacy
    raise EnvironmentError(
        "Please set MODEL_ID (preferred) or ALPACA_WEIGHTS_FOLDER in your .env. "
        "For Mistral, try MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3"
    )


def load_llm_model(device):
    """Load the LLM model and tokenizer and put the model on the specified device.

    Returns: (llm_model, tokenizer)
    """
    model_id_or_path = _resolve_model_id_or_path()
    llm_model = transformers.AutoModelForCausalLM.from_pretrained(model_id_or_path).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path)
    return llm_model, tokenizer

def add_steering_layers(llm_model, insertion_layers):
    """Add steering layers to the model.
    
    :return: llm_model
    """
    # create the steering layers
    for layer in insertion_layers:
        llm_model.model.layers[layer].mlp = SteeringLayer(llm_model.model.layers[layer].mlp)
    
    return llm_model


def load_llm_model_with_insertions(device, insertion_layers):
    """Load the llm model, the tokenizer and insert the steering layers into the model.

    :return: llm_model, tokenizer
    """
    
    llm_model, tokenizer = load_llm_model(device)
    llm_model = add_steering_layers(llm_model, insertion_layers)
    
    return llm_model, tokenizer
    