from load_lora import load_lora

model, tokenizer = load_lora()

merge_path="censor-merge"

merged_model = model.merge_and_unload()
merged_model.save_pretrained(merge_path)
tokenizer.save_pretrained(merge_path)
