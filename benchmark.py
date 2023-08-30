import argparse
from datetime import datetime
import logging
import pathlib

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch as t

import gpt2_ai.benchmark as bm
import gpt2_ai.benchmark.benchmark_factory as bf


LOG_DIR = pathlib.Path(__file__).parents[0] / 'logs' / "benchmark"


if __name__ == "__main__":

    log_file = datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    log_file = LOG_DIR / log_file
    logging.basicConfig(filename=log_file, level=logging.INFO)


    benchmark = bf.get_benchmark(bf.BenchmarkType.LAMBADA)
    
    device = 'gpu' if t.cuda.is_available() else 'cpu'
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ds = bm.lambada.LAMBADADataset(name="lambada", dev_mode=True)
    ds.get_dataset()

    benchmark = benchmark(name="lambada", model=model, tokenizer=tokenizer,
                          device=device, dataset=ds)
    res = benchmark.run()

    logging.info(f"Lambada: {res}")