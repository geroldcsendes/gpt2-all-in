import argparse

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch as t

import gpt2_ai.benchmark as bm
import gpt2_ai.benchmark.benchmark_factory as bf


if __name__ == "__main__":

    benchmark = bf.get_benchmark(bf.BenchmarkType.LAMBADA)
    
    device = 'gpu' if t.cuda.is_available() else 'cpu'
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    

    benchmark = benchmark(name="lambada", model=model, tokenizer=tokenizer, device=device)
    benchmark.get_dataset()
    benchmark.run()
    benchmark.debug_examples()