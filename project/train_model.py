



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, AutoModel, TrainingArguments, DataCollatorWithPadding, Trainer
from datasets import load_dataset
import torch


save_dir = "./finetuned_model"
model_dir = "DeepSeek-R1-Distill-Qwen-1.5B" 

def tokenize_dataset(dataset):
    tokenized = tokenizer(dataset["text"], padding="max_length", truncation=True)
    return tokenized


tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto"
)


dataset = load_dataset("rotten_tomatoes")
dataset = dataset.map(tokenize_dataset, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# config = AutoConfig.from_pretrained(model_dir)
# model = AutoModel.from_config(config)

training_args = TrainingArguments(
    output_dir=save_dir,
    learning_rate=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    eval_strategy="epoch",
    # use_cpu=True,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)


trainer.train()
