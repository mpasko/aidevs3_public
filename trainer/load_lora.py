from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

lora_path = "censor-training/checkpoint-2000"

def load_lora():
    config = PeftConfig.from_pretrained(lora_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer