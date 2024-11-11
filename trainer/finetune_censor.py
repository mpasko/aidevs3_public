from transformers import AutoModelForCausalLM, GPT2TokenizerFast, TrainingArguments
import torch

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from dataset import load_tokenize_split
from utils import print_number_of_trainable_model_parameters

model_name = 'sdadas/polish-gpt2-small'

original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
original_model.config.use_cache = False
print(print_number_of_trainable_model_parameters(original_model))

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

train_dataset, eval_dataset = load_tokenize_split("./censor_samples.txt", tokenizer)

lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    lora_dropout = 0.05,
    bias='none',
    use_rslora=True,
)

peft_model = get_peft_model(original_model, lora_config)

print(print_number_of_trainable_model_parameters(peft_model))

output_dir = f'./censor-training'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=3e-5,
    per_device_train_batch_size=100,
    num_train_epochs=1,
    logging_steps=10,
    max_steps=2000,
    report_to='none'
)

peft_trainer = SFTTrainer(
    model=peft_model, 
    args=peft_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

peft_trainer.train()

peft_model_path = './censor-checkpoint-local'

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
