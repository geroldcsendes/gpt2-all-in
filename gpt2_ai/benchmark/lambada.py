"""
Eval models on the LAMBADA dataset.
Accuracy results based on the GPT2 paper (https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):
    GPT2-small:  45.99
    GPT2-medium: 55.48
    GPT2-large:  60.12
    GPT2-xl:     63.24
Generally, this evaluation is very messy because of a number of reasons:
- OpenAI used their own, reformatted and cleaned version of the dataset
- the target words may be represented by multiple tokens. It is not clear how to calculate NLL in this case
- the exact decoding procedure is not specified

A few threads on the topic:
- https://github.com/EleutherAI/lm-evaluation-harness/issues/350
- https://github.com/openai/gpt-2/issues/131
- https://github.com/huggingface/transformers/issues/491

The openai version of the dataset is available here: https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl.
This is to be found at benchmark/data/lambada_test.jsonl.
"""

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch as t
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from random import sample
import pathlib
from string import punctuation

from gpt2_ai.benchmark.base import BaseBenchmark, BaseDataset, dev_mode_sample


LAMBADA_DIR = pathlib.Path(__file__).parents[2]
LAMBADA_PATH = LAMBADA_DIR / 'data' / 'benchmark' / 'lambada_test.jsonl'


class LAMBADADataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_path = LAMBADA_PATH

    def get_dataset(self):

        @dev_mode_sample
        def f(self):
            lambada_data = []
            with open(self.data_path) as f:
                for line in f:
                    lambada_data.append(json.loads(line)['text'])

            return lambada_data
        
        dataset = f(self)
        self.dataset = dataset

        return
    
    def get_batch(self, batch_size: int):
        # todo add batching functionality. Right now batch_size 1 is used
        raise NotImplementedError 


class LAMBADA(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def debug_examples(self, ground_truth:str, pred:str, prompt:str) -> None:
        print(f"Showing {self.debug_sample_num} random mistaken predictions")
        print("Format: ground truth - prediction", '\n' + '-'*50)
        rnd_idx = sample(range(len(ground_truth)), k=self.debug_sample_num)

        for element in rnd_idx:
            print('Prompt:\n')
            print(' '.join(prompt[element]), '\n')
            print(ground_truth[element], ' - ', pred[element])
            print('-'*50, '\n')
        return
        
    def run(self):

        res = []
        target_list = []
        out_list = []

        for sample in tqdm(self.dataset.dataset):  #lambada_data
            
            # target is the last word in the example
            target = sample.split(' ')[-1]

            # input is everything preceeding the last word
            sample_inp = ' '.join(sample.split(' ')[:-1])

            input_ids = self.tokenizer(
                sample_inp, return_tensors="pt",
                return_attention_mask=False)['input_ids']
            
            # target may be represented by multiple tokens
            target_ids = self.tokenizer(
                target, return_tensors="pt",
                return_attention_mask=False)['input_ids']

            target_token_len = target_ids.shape[1]

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # TODO: add switch to use other decoding methods
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=target_token_len,
                num_beams=5,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # we only care about the last n tokens where n is the number of tokens
            # in to represent the target word
            decoded_list = self.tokenizer.batch_decode(
                outputs.sequences[0, -target_token_len:])  # 0 is the batch index
            
            # remove whitespace and punctuation
            # in some examples, the target word is correctly predicted but followed
            # by a punctuation mark or space e.g. James vs James., water vs water,
            # seems to make sense to accept these as correct predictions
            out = ''.join(decoded_list).strip()
            out = out.strip(punctuation)

            #loss = criterion(logits, target_ids.squeeze(0))
            res.append(out == target)
            target_list.append(target)
            out_list.append(out)

        acc = sum(res) / len(res)
        print(f'Accuracy: {acc:.2f} in {len(res)} examples')

        if self.debug_sample_num > 0:
            mistake_idx = [cnt for cnt, element in enumerate(res) if not element]
            ground_truth = [element for cnt, element in enumerate(target_list) if cnt in mistake_idx]
            pred = [element for cnt, element in enumerate(out_list) if cnt in mistake_idx]
            prompt = [element.split(' ')[:-1] for cnt, element in enumerate(self.dataset.dataset) if cnt in mistake_idx]
            self.debug_examples(ground_truth=ground_truth, pred=pred, prompt=prompt)

        return acc

        
    
