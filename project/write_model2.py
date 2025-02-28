



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import argparse

model_dir = "models/DeepSeek-R1-Distill-Qwen-1.5B"
finetuned_model_dir = "models/Go_model" 


tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto"
    
)

model = PeftModel.from_pretrained(model, finetuned_model_dir)
def commands():
    parser = argparse.ArgumentParser()
    parser.add_argument("quistion", help="quistion to ask ai")
    args = parser.parse_args()
    return args

model = model.merge_and_unload()

def response(quistion):
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    resault = text_generator(quistion, max_length=512, temperature=0.5, truncation=True)
    return resault

if __name__ == "__main__":
    commands()
    response = response(commands().quistion)
    print(response[0]['generated_text'])