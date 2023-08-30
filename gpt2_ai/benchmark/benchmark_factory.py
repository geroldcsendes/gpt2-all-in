from enum import Enum
from gpt2_ai.benchmark.lambada import LAMBADA
from gpt2_ai.benchmark.base import BaseBenchmark

class BenchmarkType(Enum):
    """Enum class for benchmark types."""

    # Benchmark types
    BENCHMARK_ALL = "benchmark_all"
    LAMBADA = "lambada"
    WIKITEXT103 = "wikitext103"


def get_benchmark(benchmark_type: BenchmarkType) -> BaseBenchmark:
    """
    Factory function for choosing a benchmark type.
    """

    if benchmark_type == BenchmarkType.LAMBADA:
        return LAMBADA
    # elif benchmark_type == BenchmarkType.WIKITEXT103:
    #     return bm.WikiText103
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

