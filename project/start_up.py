from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import os

# Print GPU information
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Define model directories
base_model_dir = "./models/DeepSeek-R1-Distill-Qwen-1.5B"
lora_model_dir = "./finetuned_lora_model"

# Load tokenizer
print(f"Loading tokenizer from {base_model_dir}")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    print("Base tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading base tokenizer: {e}")
    print("Trying to load tokenizer from adapter directory...")
    tokenizer = AutoTokenizer.from_pretrained(lora_model_dir, trust_remote_code=True)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare for model loading
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Try to load base model and LoRA adapter
try:
    # First load the base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Get PEFT configuration
    print("Getting PEFT configuration...")
    peft_config = PeftConfig.from_pretrained(lora_model_dir)
    
    # Then load the LoRA adapter with more explicit handling
    print(f"Loading LoRA adapter from {lora_model_dir}")
    
    # Try with specific adapter name
    try:
        model = PeftModel.from_pretrained(
            base_model, 
            lora_model_dir,
            adapter_name="default"
        )
    except TypeError:
        # Fall back to simpler loading without extra parameters
        model = PeftModel.from_pretrained(base_model, lora_model_dir)
    
    print("Model with LoRA adapter loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative loading method...")
    
    try:
        # Fall back to just using the base model if adapter loading fails
        print("Falling back to base model only...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            device_map="auto", 
            trust_remote_code=True
        )
        print("Base model loaded successfully as fallback!")
    except Exception as e2:
        print(f"Error loading fallback model: {e2}")
        print("Exiting...")
        exit(1)

# Simple interactive loop
def generate_response(prompt):
    print(f"Generating response for: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Use generation parameters from generation_config.json
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            max_length=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.95
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Model ready! Enter prompts (or 'exit' to quit):")
    print("Try asking questions about Go programming!")
    while True:
        prompt = input("> ")
        if prompt.lower() in ["exit", "quit"]:
            break
        try:
            response = generate_response(prompt)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error generating response: {e}")