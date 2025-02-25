from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "project/DeepSeek-R1-Distill-Qwen-1.5B" 

tokenizer = AutoTokenizer.from_pretrained(model_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
).to(device)
is_running = True
while is_running:
    user_input = input("Enter your prompt: ")
    print(f"You Asked: {user_input}")
    if user_input == "exit":
        is_running = False
        break
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(device)
    print("Loading...")
    output = model.generate(input_ids, max_length=200, top_p=0.95)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
