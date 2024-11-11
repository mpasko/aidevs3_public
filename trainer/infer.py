

def generate(text, model, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt').to("cuda")
    output = model.generate(**encoded_input, max_length=400)
    chunks = [tokenizer.decode(output[x]) for x in range(len(output))]
    return "".join(chunks)

def interactive(model, tokenizer):
    while True:
        text = input()
        output = generate(text, model, tokenizer)
        print(output)