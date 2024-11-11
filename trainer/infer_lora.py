
from infer import interactive
from load_lora import load_lora

model, tokenizer = load_lora()

interactive(model, tokenizer)
