"""The single inference abstraction seam (design 5.1).

Both the mock backend and the Modal/vLLM client implement :class:`InferenceBackend`
and return the identical :class:`Completion` type. The env and agents never branch on
which backend is in use; that is what makes the whole pipeline GPU-free-testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Completion:
    """A single chat completion result.

    ``seed`` and ``temperature`` are echoed back so callers can confirm the request
    parameters that produced the text (useful for determinism auditing).
    """

    text: str
    model: str
    finish_reason: str
    seed: int
    temperature: float


class InferenceBackend(ABC):
    """Abstract chat backend. Identical surface for mock and real implementations."""

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> Completion:
        """Return a :class:`Completion` for ``messages`` using ``model``."""

    @abstractmethod
    def available_models(self) -> list[str]:
        """Return the list of model identifiers this backend can serve."""
