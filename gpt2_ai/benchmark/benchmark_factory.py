from enum import Enum
from typing import Union

from gpt2_ai.benchmark.lambada import LAMBADA, LAMBADADataset
from gpt2_ai.benchmark.cbt import CBT, CBTDataset
from gpt2_ai.benchmark.wikitext import WikiText103, WikiText103Dataset
from gpt2_ai.benchmark.base import BaseBenchmark, BaseDataset


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
