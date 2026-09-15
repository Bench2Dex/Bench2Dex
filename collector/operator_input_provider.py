"""Teleop-agnostic operator input provider interface."""

from __future__ import annotations

from typing import Protocol


class OperatorInputProvider(Protocol):
    """Protocol for operator input providers (e.g., teleop, keyboard)."""

    def start(self) -> bool:
        """Start the provider. Returns True if started successfully."""
        ...

    def stop(self) -> None:
        """Stop the provider."""
        ...

    def describe(self) -> dict:
        """Return provider metadata.

        Returns:
            dict with keys:
                - provider_name: str identifier
                - operator_available: bool
        """
        ...

    def snapshot(self) -> dict | None:
        """Capture current operator input snapshot.

        Returns:
            dict with operator input data, or None if not available.
        """
        ...


class NullOperatorInputProvider:
    """Null provider that returns no operator input."""

    def start(self) -> bool:
        """Always returns False - no provider available."""
        return False

    def stop(self) -> None:
        """No-op."""
        return None

    def describe(self) -> dict:
        """Return null provider description."""
        return {"provider_name": "none", "operator_available": False}

    def snapshot(self) -> dict | None:
        """Always returns None - no operator input."""
        return None