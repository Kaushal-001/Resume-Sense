import os
import platform
from huggingface_hub import hf_hub_download

def download_models():
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    system = platform.system()
    processor = platform.processor()

    print(f"🖥️  Detected System: {system} ({processor})")

    if system == "Darwin" and "arm" in processor.lower():
        print("🍏 Apple Silicon detected. Downloading GGUF model for Metal acceleration...")
        
        # Repo and filename for GGUF (Quantized Phi-3)
        repo_id = "QuantFactory/Phi-3-mini-4k-instruct-GGUF"
        filename = "Phi-3-mini-4k-instruct.Q4_K_M.gguf"
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"✅ GGUF Model downloaded to: {local_path}")
        
    else:
        print("☁️  Linux/CUDA/Windows detected. The pipeline will download the main model automatically via Transformers.")

if __name__ == "__main__":
    download_models()