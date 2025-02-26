



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, AutoModel, TrainingArguments, DataCollatorWithPadding, Trainer
from datasets import load_dataset
import torch

# model_dir = "distilgpt2"
model_dir = "finetuned_model/model" 


tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto"
    
)

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
resault = text_generator("what is go lang", max_length=25, temperature=0.1, truncation=True)
print(resault)
# config = AutoConfig.from_pretrained(model_dir)
# model = AutoModel.from_config(config)