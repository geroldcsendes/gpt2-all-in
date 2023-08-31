from enum import Enum
from typing import Union

from gpt2_ai.benchmark.lambada import LAMBADA, LAMBADADataset
from gpt2_ai.benchmark.cbt import CBT, CBTDataset
from gpt2_ai.benchmark.wikitext import WikiText103, WikiText103Dataset
from gpt2_ai.benchmark.base import BaseBenchmark, BaseDataset
from gpt2_ai.benchmark.models import GPT2, GPTNeo, ModelType


class BenchmarkType(str, Enum):
    """Enum class for benchmark types."""

    # Benchmark types
    BENCHMARK_ALL = "benchmark_all"
    LAMBADA = "lambada"
    CBT = "cbt"
    WIKITEXT103 = "wikitext103"


def get_benchmark(benchmark_type: str) -> Union[BaseBenchmark, BaseDataset]:
    """
    Factory function for choosing a benchmark type.
    """

    if benchmark_type == BenchmarkType.LAMBADA:
        return LAMBADA, LAMBADADataset
    elif benchmark_type == BenchmarkType.CBT:
        return CBT, CBTDataset
    elif benchmark_type == BenchmarkType.WIKITEXT103:
        return WikiText103, WikiText103Dataset
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


def get_model(model_type: str):
    """
    Factory function for choosing a model type.
    """
    if model_type == ModelType.GPT2:
        return GPT2()
    elif model_type == ModelType.GPT_NEO_SMALL:
        return GPTNeo(ModelType.GPT_NEO_SMALL)
    # todo add other options
    else:
        model_options = [_.value for _ in ModelType]
        model_options = ", ".join(model_options)
        raise ValueError(f"Unknown model type: {model_type}. Please choose from: {model_options}")
