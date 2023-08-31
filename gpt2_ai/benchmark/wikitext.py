"""
Eval models on WikiText dataset. In the GPT2 paper, they evaluated the models on Wiki datasets: the WikiText2 and enwiki8 datasets.
I think the WikiText2 dataset (said to be ca. 2m rows) is the same as the wikitext-103-raw-v1 on [Hugginface](https://huggingface.co/datasets/wikitext)
Perplexity results on WikiText2 as per the paper
    GPT2-small:  29.41
    GPT2-medium: 22.76
    GPT2-large:  19.93
    GPT2-xl:     18.34
"""

from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader
import torch as t
from tqdm import tqdm

from gpt2_ai.benchmark.base import BaseDataset, BaseBenchmark, dev_mode_sample


class WikiText103Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_batch(self, batch_size: int):
        raise NotImplementedError
    
    def get_dataset(self):
        
        @dev_mode_sample
        def f(self):
            ds =  load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            return  ds
        
        dataset = f(self)

        self.dataset = dataset

        return
    

class WikiText103(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def debug_examples(self, **kwargs):
        pass

    def encode(self, example):
        # TODO: parameterize max_length
        return self.tokenizer(
            example['text'], truncation=True, padding='max_length',
            max_length=1024,
            return_tensors='np', return_attention_mask=False)
    

    def run(self):
        # filter section names and empty lines
        ds = self.dataset.dataset.filter(lambda example: example['text'] != "" and not example['text'].startswith(" ="))
        ds = ds.map(self.encode, batched=True, remove_columns=['text'])
        ds = ds.with_format("torch")

        # TODO: parameterize max length
        loader = DataLoader(ds, batch_size=2, num_workers=4)
        self.model.eval()

        losses = []
        for sample in tqdm(loader):
            sample = sample['input_ids'].to(self.device)
            
            # set padding tokens to -100 to ignore them in loss calculation
            labels = t.clone(sample)
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            with t.inference_mode():
                out = self.model(sample, labels=labels, use_cache=False)

            loss = out.loss.detach().cpu().item()
            losses.append(loss)

        ce_avg = np.mean(losses)
        ppl = np.exp(ce_avg)
        print(f"Cross-entropy: {ce_avg:.2f}")
        print(f"Perplexity: {ppl:.2f}")

        return ppl

