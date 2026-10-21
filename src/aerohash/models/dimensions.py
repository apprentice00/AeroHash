"""Immutable GAP hash dimensions shared by the model and optimization."""
from dataclasses import dataclass
from .validation import positive_integer


@dataclass(frozen=True)
class HashDimensions:
    requested_length: int
    training_dim: int
    retrieval_dim: int


def resolve_dimensions(code_length):
    """GAP emits exactly the requested bits in training and evaluation."""
    positive_integer('code_length', code_length)
    return HashDimensions(code_length, code_length, code_length)
