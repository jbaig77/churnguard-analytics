"""Base agent interface.

All agents are thin wrappers that delegate heavy work to core modules.
They communicate through the Orchestrator's shared state, not direct
imports of each other — keeping them loosely coupled.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class that every domain agent must subclass.

    Provides a common constructor, a mandatory message-dispatch hook, and two
    convenience helpers for building uniform response dicts.  Concrete agents
    only need to implement :meth:`handle`; all other behaviour is inherited.
    """

    def __init__(self, name: str):
        """Initialise the agent and attach a named logger.

        Args:
            name: Short identifier for this agent (e.g. ``"analytics"``).
                  Used as the logger name suffix and as the registration key
                  inside :class:`~agents.orchestrator.Orchestrator`.
        """
        self.name = name
        self.logger = logging.getLogger(f"agents.{name}")

    @abstractmethod
    def handle(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an inbound message and return a structured response.

        Concrete subclasses must implement this method and branch on
        ``msg_type`` to call the appropriate private handler.

        Args:
            msg_type: String token that identifies the requested operation
                      (e.g. ``"summary"``, ``"train_or_load"``).
            payload:  Arbitrary key/value arguments specific to the message
                      type.  May be an empty dict when the operation requires
                      no parameters.

        Returns:
            Dict containing at minimum ``{"status": "ok"}`` on success or
            ``{"status": "error", "message": <str>}`` on failure, plus any
            additional keys populated by the handler.
        """

    def _ok(self, **kwargs) -> dict[str, Any]:
        """Build a success response dict, merging any extra keyword arguments.

        Args:
            **kwargs: Arbitrary additional keys to include in the response
                      alongside ``status``.

        Returns:
            Dict with ``"status"`` set to ``"ok"`` and every supplied keyword
            argument as an extra top-level key.
        """
        return {"status": "ok", **kwargs}

    def _err(self, message: str) -> dict[str, Any]:
        """Log an error and build a failure response dict.

        Args:
            message: Human-readable description of what went wrong.  Logged at
                     ERROR level via this agent's named logger.

        Returns:
            Dict with ``"status"`` set to ``"error"`` and ``"message"`` set to
            the supplied string.
        """
        self.logger.error(message)
        return {"status": "error", "message": message}
