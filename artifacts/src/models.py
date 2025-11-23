import os
import platform
import torch
from langchain_huggingface import HuggingFacePipeline

def load_llm():
    """
    Universal LLM Loader.
    - Detects Apple Silicon (M1/M2) -> Loads GGUF via LlamaCpp (Metal GPU)
    - Detects Linux/NVIDIA -> Loads BitsAndBytes Quantized Model (CUDA)
    """
    system = platform.system()
    processor = platform.processor()

    # --- Option 1: Apple Silicon (M1/M2) with GGUF via LlamaCpp ---
    if system == "Darwin" and processor == "arm":
        print("🍎 Detected Mac M1/M2 - Loading GGUF Model with Metal Support")

        try:
            from langchain_community.llms import LlamaCpp
            from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
        except ImportError:
            raise ImportError("Please install llama-cpp-python. Run: CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python")
        
        model_path = "models/Phi-3-mini-4k-instruct.Q4_K_M.gguf"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run 'python setup_model.py' first.")
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=1,
            n_batch=512,
            temperature=0.1,
            max_tokens=2000,
            n_ctx=4096,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=True
        )
        return llm
    
    # --- Option 2: Linux/NVIDIA with BitsAndBytes Quantized Model ---
    else:
        print("🖥️  Detected Linux/NVIDIA - Loading BitsAndBytes Quantized Model with CUDA Support")

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        model_name = "microsoft/Phi-3-mini-4k-instruct"

        # Configure 4-bit quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        # Create HuggingFace Pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,
            temperature=0.1,
            do_sample=True,
            return_full_text=False # LangChain expects only the generated text
        )
        return HuggingFacePipeline(pipeline=pipe)