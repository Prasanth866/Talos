"""Math and calculation module."""

from dataclasses import dataclass


def add(a: int, b: int = 0) -> int:
    """Adds two integers together."""
    return a + b


@dataclass
class Calculator:
    """A standard arithmetic calculator class."""

    precision: int = 2

    def multiply(self, x: float, y: float) -> float:
        """Multiplies two floating point numbers."""
        return round(x * y, self.precision)

    @classmethod
    def create_default(cls) -> Calculator:
        """Factory method to create a default calculator."""
        return cls(precision=4)
