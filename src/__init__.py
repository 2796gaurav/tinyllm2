"""TinyLLM Guardrail: Sub-100MB LLM Security System"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.models.dual_branch import TinyGuardrail
from src.models.embeddings import ThreatAwareEmbedding
from src.training.trainer import GuardrailTrainer
from src.evaluation.benchmarks import GuardrailBenchmark

__all__ = [
    "TinyGuardrail",
    "ThreatAwareEmbedding",
    "GuardrailTrainer",
    "GuardrailBenchmark",
]

