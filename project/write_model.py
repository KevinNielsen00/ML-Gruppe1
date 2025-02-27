



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, AutoModel, TrainingArguments, DataCollatorWithPadding, Trainer
from datasets import load_dataset
from peft import PeftModel
import torch

model_dir = "models/DeepSeek-R1-Distill-Qwen-1.5B"
finetuned_model_dir = "finetuned_lora_model" 


tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto"
    
)

model = PeftModel.from_pretrained(model, finetuned_model_dir)


model = model.merge_and_unload()

def quistion():
    print("What is your question?")
    question = input()
    return question

def response(quistion):
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    resault = text_generator(quistion, max_length=512, temperature=0.5, truncation=True)
    return resault

while True:
    a_question = quistion()
    if a_question.lower == "exit":
        break
    print(f"Your question is: {a_question}")
    a_response = response(a_question)
    print(f"Answer: {a_response[0]['generated_text']}")

    

