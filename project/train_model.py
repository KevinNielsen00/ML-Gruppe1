from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import torch
import json

# Define directories
save_dir = "./finetuned_lora_model"
model_dir = "models/DeepSeek-R1-Distill-Qwen-1.5B" 

# Load your custom dataset
def load_custom_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the text data
    texts = [item['answer'] for item in data]
    
    # Create a dataset dictionary
    dataset_dict = {"text": texts}
    
    # Convert to Dataset object
    return Dataset.from_dict(dataset_dict)

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    quantization_config=quantization_config
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def tokenize_dataset(dataset):
    return tokenizer(
        dataset["text"],
        padding="max_length", 
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True
    )

dataset = load_custom_dataset("datacenter/data/dataset.json")
tokenized_dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

training_args = TrainingArguments(
    learning_rate=3e-5,
    output_dir=save_dir,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  
    num_train_epochs=3,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    weight_decay=0.01,
    warmup_steps=50,
    fp16=True,
    optim="adamw_torch",
    max_grad_norm=0.3,
    report_to=["none"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"LoRA adapter saved to {save_dir}")
