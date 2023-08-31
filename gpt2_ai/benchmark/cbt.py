"""
Eval models on the Children's Book Test (CBT) dataset.
Accuracy results based on Named Entity (NE) subtask in the GPT2 paper (https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):
    GPT2-small:  83.04
    GPT2-medium: 87.10
    GPT2-large:  88.00
    GPT2-xl:     89.05

Original paper: https://arxiv.org/pdf/1511.02301v4.pdf
"""

from datasets import load_dataset
import numpy as np
import torch as t
from tqdm import tqdm

from gpt2_ai.benchmark.base import BaseDataset, BaseBenchmark, dev_mode_sample


# the prompt format below may be off. The paper does not specify it and I found no other
# credible implementations
PROMPT_FMT = "Fill in the blank in the last sentence.\n{sentence}\n{question}\n"


class CBTDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_batch(self, batch_size: int):
        # todo add batching functionality. Right now batch_size 1 is used
        raise NotImplementedError 

    def get_dataset(self):
        
        @dev_mode_sample
        def f(self):
            ds = load_dataset('cbt', 'NE', split='test')
            return  ds
    
        dataset = f(self)
        self.dataset = dataset

        return
    

class CBT(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def debug_examples(self, **kwargs):
        pass

    @staticmethod
    def detokenize(text):
        text = text.replace(" '", "'")
        text = text.replace(" \n", "\n")
        text = text.replace("\n ", "\n")
        text = text.replace(" n't", "n't")
        text = text.replace("`` ", '"')
        text = text.replace("''", '"')
        # punctuation
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        return text

    def run(self):
        res = []
        self.model.eval()
        for sample in tqdm(self.dataset.dataset):
            text = PROMPT_FMT.format(sentence="\n".join(sample['sentences']), question=sample["question"])
            logits = []
            
            for option in sample["options"]:
                # text_inp = text.replace("XXXXX", option)
                text_inp = text + " " + option
                text_inp = self.detokenize(text_inp)
                inp_ids = self.tokenizer(text_inp, return_tensors="pt", return_attention_mask=False)
                inp_ids.to(self.device)

                with t.inference_mode():
                    out = self.model(**inp_ids)
                logit = out.logits[0][-1].max().detach().cpu().item()
                logits.append(logit)

            corr = np.argmax(logits) == sample['options'].index(sample['answer'])
            res.append(corr)

        acc = sum(res) / len(res)
        print(f'Accuracy: {acc:.2f} in {len(res)} examples')

        return acc