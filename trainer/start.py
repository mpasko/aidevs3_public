from transformers import GPT2Tokenizer, GPT2LMHeadModel

from infer import interactive

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('./censor-merge/', local_files_only=True)
model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('./censor-merge/').to("cuda")

interactive(model, tokenizer)
