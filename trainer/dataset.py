from transformers import GPT2TokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, RandomSampler, SequentialSampler

max_length = 180

def load_dataset(filename: str):
    with open(filename) as file:
        return file.read().split("\n")

def format_example(example: str):
    if "<s>" not in example:
        return None
    replacement = example
    wrapped = f"{replacement}</s>"
    return wrapped

def load_and_format(filename: str):
    raw_data = load_dataset(filename)
    formatted = [ format_example(token) for token in raw_data ]
    correct = [ token for token in formatted if token is not None ]
    return correct

def split_dataset(dataset):
    split_size = int(len(dataset)*0.9)
    train_dataset = dataset[0: split_size]
    eval_dataset = dataset[split_size+1:]
    return train_dataset, eval_dataset

def load_tokenize_split(filename: str, tokenizer: GPT2TokenizerFast):
    raw_data = load_and_format(filename)
    train, test = split_dataset(raw_data)
    train_dataset = CensorDataset(train, tokenizer)
    test_dataset = CensorDataset(test, tokenizer)
    return train_dataset, test_dataset

def tokenize_default(example: str, tokenizer: GPT2TokenizerFast):
    encodings = tokenizer(
        example,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'       
    )
    return encodings

# standard PyTorch approach of loading data in using a Dataset class.
class CensorDataset(Dataset):
    def __init__(self, data, tokenizer: GPT2TokenizerFast):
        self.data = data
        self.input_ids = []
        self.attn_masks = []
        for example in data:
            encodings = tokenize_default(example, tokenizer)
            self.input_ids.append(torch.squeeze(encodings['input_ids']))
            self.attn_masks.append(torch.squeeze(encodings['attention_mask']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return {
            'input_ids': self.input_ids[idx],
        }