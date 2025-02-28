from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    GPT2LMHeadModel, 
    GPT2TokenizerFast
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import torch
import json
import os
import pandas as pd
import glob
from tqdm import tqdm
import multiprocessing
import random

os.makedirs("./logs", exist_ok=True)
os.makedirs("./finetuned_gpt2_go", exist_ok=True)

save_dir = "./finetuned_gpt2_go"
model_dir = "distilgpt2"

def load_custom_dataset(data_dir="datacenter", max_samples=20000):
    texts = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist")
        return Dataset.from_dict({"text": []})
    
    print(f"Loading data from {data_dir}")
    
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    
    print(f"Found {len(json_files)} JSON files and {len(csv_files)} CSV files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = 0
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            content = item.get('answer', '') or item.get('content', '')
                            if 'go' in str(content).lower() or 'func ' in str(content) or 'package ' in str(content):
                                if len(str(content)) > 20:
                                    texts.append(content)
                                    count += 1
                print(f"Loaded {count} Go examples from {json_file}")
        except Exception as e:
            print(f"Warning: Error loading {json_file}: {e}")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            count = 0
            for col in ['content', 'answer', 'response', 'code', 'text', 'body']:
                if col in df.columns:
                    for text in df[col].dropna().tolist():
                        if 'go' in str(text).lower() or 'func ' in str(text) or 'package ' in str(text):
                            if len(str(text)) > 20:
                                texts.append(str(text))
                                count += 1
            print(f"Loaded {count} Go examples from CSV: {csv_file}")
        except Exception as e:
            print(f"Warning: Error loading {csv_file}: {e}")
    
    go_examples = [
        "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"Hello, Go!\")\n}",
        "func bubbleSort(arr []int) []int {\n\tn := len(arr)\n\tfor i := 0; i < n; i++ {\n\t\tfor j := 0; j < n-i-1; j++ {\n\t\t\tif arr[j] > arr[j+1] {\n\t\t\t\tarr[j], arr[j+1] = arr[j+1], arr[j]\n\t\t\t}\n\t\t}\n\t}\n\treturn arr\n}"
    ]
    
    texts.extend(go_examples)
    
    texts = list(set(texts))
    print(f"After removing duplicates: {len(texts)} examples")
    
    if max_samples and len(texts) > max_samples:
        random.shuffle(texts)
        texts = texts[:max_samples]
    
    texts = [text for text in texts if text and len(text) > 20]
    
    print(f"Total training examples: {len(texts)}")
    dataset_dict = {"text": texts}
    
    return Dataset.from_dict(dataset_dict)

def tokenize_dataset(dataset):
    return tokenizer(
        dataset["text"],
        padding="max_length", 
        truncation=True,
        max_length=256,
        return_special_tokens_mask=True
    )

def main():
    global tokenizer
    
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added [PAD] token to tokenizer")

    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    
    model.resize_token_embeddings(len(tokenizer))
    
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    model.train()
    
    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        elif 'bias' in name:
            param.requires_grad = True
    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )
    
    print("Loading datasets...")
    dataset = load_custom_dataset(data_dir="datacenter", max_samples=30000)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Creating train/test splits...")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.05)
    print(f"Train examples: {len(tokenized_dataset['train'])}")
    print(f"Test examples: {len(tokenized_dataset['test'])}")

    training_args = TrainingArguments(
        learning_rate=2e-5,
        output_dir=save_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    torch.set_num_threads(4)

    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        print("Trying alternative manual training approach...")
        
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=1e-5
        )
        
        model.train()
        num_manual_epochs = 3
        print(f"Starting manual training for {num_manual_epochs} epochs")
        
        for epoch in range(num_manual_epochs):
            total_loss = 0
            batch_count = 0
            for batch in tqdm(trainer.get_train_dataloader(), desc=f"Manual epoch {epoch+1}"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_count % 50 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}, Avg Loss: {total_loss/batch_count:.4f}")
            
            print(f"Epoch {epoch+1} complete, Average Loss: {total_loss/batch_count:.4f}")

    print("Saving model...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("Creating merged model for easier inference...")
    try:
        merged_model = model.merge_and_unload()
        merged_save_dir = os.path.join(save_dir, "merged")
        os.makedirs(merged_save_dir, exist_ok=True)
        merged_model.save_pretrained(merged_save_dir)
        print(f"Merged model saved to {merged_save_dir}")
    except Exception as e:
        print(f"Warning: Could not save merged model: {e}")

    print(f"DistilGPT-2 Go Coding model saved to {save_dir}")
    
    test_prompt = "Write a Go function that sorts a slice of integers using bubble sort"
    
    print("\nTesting the model with prompt:")
    print(test_prompt)
    
    device = model.device
    
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=500,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated Go code:")
    print(generated_text)

    try:
        with open("generated_go_code.go", "w") as f:
            f.write(generated_text)
        print("Generated code saved to 'generated_go_code.go'")
    except Exception as e:
        print(f"Could not save generated code to file: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()