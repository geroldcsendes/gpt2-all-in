import argparse
from datetime import datetime
import logging
import pathlib

import torch as t

import gpt2_ai.benchmark.benchmark_factory as bf


LOG_DIR = pathlib.Path(__file__).parents[0] / 'logs' / "benchmark"


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=[_.value for _ in bf.BenchmarkType])
    parser.add_argument("--model" , type=str, choices=[_.value for _ in bf.ModelType],
                        default=bf.ModelType.GPT2.value)
    parser.add_argument("--dev", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    #benchmark = bf.get_benchmark(bf.BenchmarkType.LAMBADA)
    benchmark, ds = bf.get_benchmark(args.task)

    log_file = datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    log_file = LOG_DIR / log_file
    logging.basicConfig(filename=log_file, level=logging.INFO)

    ds = ds(dev_mode=args.dev)
    ds.get_dataset()
    print('Ds: ', ds.dataset[:10])

    device = 'gpu' if t.cuda.is_available() else 'cpu'

    model = bf.get_model(args.model)

    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model.config.pad_token_id = model.config.eos_token_id
    # model.generation_config.pad_token_id = model.config.eos_token_id
    # model.to(device)

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.pad_token = tokenizer.eos_token

    benchmark = benchmark(model=model.model, tokenizer=model.tokenizer,
                          device=device, dataset=ds)
    res = benchmark.run()

    logging.info("Lambada: %s", res)
