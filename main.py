from huggingface_hub import snapshot_download
from pathlib import Path

# Set a directory to save the model weights and config
model_dir = Path.home() / 'mistral_models' / 'Pixtral'
model_dir.mkdir(parents=True, exist_ok=True)

# Download the model files (params.json, consolidated.safetensors, tekken.json)
snapshot_download(repo_id="mistralai/Pixtral-12B-2409", 
                  allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
                  local_dir=model_dir)
print(f"Model downloaded to {model_dir}")
