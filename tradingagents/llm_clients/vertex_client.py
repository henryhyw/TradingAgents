from __future__ import annotations

import os
from typing import Any, Optional

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class VertexAIClient(BaseLLMClient):
    """Client for Vertex AI Gemini models using ADC credentials."""

    provider = "vertex"

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatVertexAI instance."""
        self.warn_if_unknown_model()
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:  # pragma: no cover - depends on optional runtime install
            raise RuntimeError(
                "langchain-google-vertexai is required for provider='vertex'. "
                "Install dependencies with `pip install -e .`."
            ) from exc

        class NormalizedChatVertexAI(ChatVertexAI):
            def invoke(self, input, config=None, **kwargs):  # noqa: ANN001
                return normalize_content(super().invoke(input, config, **kwargs))

        llm_kwargs: dict[str, Any] = {}
        project = self.kwargs.get("project") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
        location = self.kwargs.get("location") or os.getenv("GOOGLE_CLOUD_REGION") or "us-central1"
        if project:
            llm_kwargs["project"] = project
        if location:
            llm_kwargs["location"] = location
        if self.kwargs.get("temperature") is not None:
            llm_kwargs["temperature"] = self.kwargs["temperature"]

        for key in ("max_retries", "timeout", "callbacks"):
            if key in self.kwargs and self.kwargs[key] is not None:
                llm_kwargs[key] = self.kwargs[key]

        # Keep optional thinking controls best-effort and fail open if unsupported.
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            llm_kwargs["thinking_budget"] = -1 if str(thinking_level).lower() == "high" else 0

        constructor_attempts = [
            ("model", {**llm_kwargs, "model": self.model}),
            ("model_name", {**llm_kwargs, "model_name": self.model}),
        ]
        for _, kwargs in constructor_attempts:
            try:
                return NormalizedChatVertexAI(**kwargs)
            except TypeError as exc:
                if "thinking_budget" in kwargs:
                    reduced = dict(kwargs)
                    reduced.pop("thinking_budget", None)
                    try:
                        return NormalizedChatVertexAI(**reduced)
                    except TypeError:
                        pass
                last_error = exc
                continue
        raise RuntimeError(
            f"Unable to initialize ChatVertexAI for model '{self.model}'. "
            f"Last error: {last_error}"
        )

    def validate_model(self) -> bool:
        return validate_model("vertex", self.model)
