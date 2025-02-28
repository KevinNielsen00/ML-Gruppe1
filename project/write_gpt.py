from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import os

model_dir = "./models/distilgpt2"
finetuned_model_dir = "./models/finetuned_gpt2_go"

if not os.path.exists(model_dir):
    print(f"Error: Base model path not found: {model_dir}")
    exit(1)
    
if not os.path.exists(finetuned_model_dir):
    print(f"Error: Fine-tuned model path not found: {finetuned_model_dir}")
    print("Available paths:")
    print("\n".join(os.listdir(".")))
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading tokenizer from {finetuned_model_dir}")
try:
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir)
    print("Successfully loaded tokenizer from fine-tuned model")
except Exception as e:
    print(f"Falling back to loading tokenizer from {model_dir}: {e}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
print(f"Tokenizer vocabulary size: {vocab_size}")

print(f"Loading base model from {model_dir}")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

print("Resizing token embeddings if needed")
model.resize_token_embeddings(len(tokenizer))

print(f"Loading fine-tuned model from {finetuned_model_dir}")
model = PeftModel.from_pretrained(model, finetuned_model_dir)

print("Merging model...")
model = model.merge_and_unload()

text_generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
)

def ask_question():
    print("\nWhat is your question? (Type 'exit' to quit)")
    question = input("> ")
    return question

def generate_response(question):
    print("Generating response...")
    result = text_generator(
        question, 
        max_length=512, 
        temperature=0.7, 
        top_p=0.95,
        do_sample=True,
        truncation=True,
        return_full_text=False
    )
    return result[0]['generated_text']

print("\n=== Go Programming Assistant ===")
print("Ask questions about Go programming or type 'exit' to quit.")

while True:
    question = ask_question()
    if question.lower() == "exit":
        print("Goodbye!")
        break
    
    print(f"\nQuestion: {question}")
    response = generate_response(question)
    print(f"\nAnswer: {response}")