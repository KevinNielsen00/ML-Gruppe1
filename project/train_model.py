



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, AutoModel, TrainingArguments, DataCollatorWithPadding, Trainer
from datasets import load_dataset, Dataset
import torch
import json



save_dir = "./finetuned_model"
model_dir = "distilgpt2" 

def tokenize_dataset(dataset):
    tokenized = tokenizer(dataset["text"], padding="max_length", truncation=True)
    return tokenized


tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length=128)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto"
    
)




# Indlæs data fra JSON-filen
with open("datacenter/dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

questions = [entry["question"] for entry in data]
answers = [entry["answer"] for entry in data]

split_ratio = 0.8
split_index = int(len(questions) * split_ratio)

train_questions = questions[:split_index]
train_answers = answers[:split_index]

eval_questions = questions[split_index:]
eval_answers = answers[split_index:]

train_encodings = tokenizer(train_questions, truncation=True, padding="max_length")
train_labels = tokenizer(train_answers, truncation=True, padding="max_length")

eval_encodings = tokenizer(eval_questions, truncation=True, padding="max_length")
eval_labels = tokenizer(eval_answers, truncation=True, padding="max_length")


train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels["input_ids"]
})
eval_dataset = Dataset.from_dict({
    "input_ids": eval_encodings["input_ids"],
    "attention_mask": eval_encodings["attention_mask"],
    "labels": eval_labels["input_ids"]
})

from transformers import DataCollatorForLanguageModeling

training_args = TrainingArguments(
    learning_rate=3e-5,
    output_dir=save_dir,
    num_train_epochs=20,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    save_total_limit=3,
    warmup_steps=1000,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

# Brug en data collator til at forbedre træningen
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # False fordi det er en causal language model (GPT-type)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Tilføj evalueringsdata
    data_collator=data_collator
)



torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

trainer.train()
trainer.save_model(f"{save_dir}/model")







# train_texts = [
#     "Flopper is a dog.",
#     "Flopper loves to play in the park.",
#     "My friend has a dog named Flopper.",
#     "Flopper barks when he sees a cat.",
#     "Flopper is a very friendly dog."
# ]

# train_labels = [
#     "Flopper is a dog.",
#     "Flopper loves to play in the park.",
#     "My friend has a dog named Flopper.",
#     "Flopper barks when he sees a cat.",
#     "Flopper is a very friendly dog."
# ]


