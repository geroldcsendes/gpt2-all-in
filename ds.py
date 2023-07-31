from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
from tqdm import tqdm


if __name__ == "__main__":
    # dataset = load_dataset("mc4", "en", streaming=True, split="train")
    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # def encode(examples):
    #     return tokenizer(examples['text'], truncation=True, padding='max_length')
    # dataset = dataset.map(encode, batched=True, remove_columns=["text", "timestamp", "url"])
    # sample = next(iter(dataset))
    # print(sample)

    #dataset = load_dataset("mc4", "en", streaming=True, split="train")
    dataset = load_dataset('oscar', 'unshuffled_deduplicated_en', streaming=True, split='train')

    dataset = dataset.rename_column("text", "input_ids")
    dataset = dataset.with_format("torch")

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def encode(examples):
        return tokenizer(examples['input_ids'], truncation=True, padding='max_length')
    dataset = dataset.map(encode, batched=True, remove_columns=["id"])

    dataloader = DataLoader(
        dataset,
        collate_fn=DataCollatorForLanguageModeling(tokenizer))

    i = 0
    for batch in dataloader:
        print(batch)
        i += 1
        break 