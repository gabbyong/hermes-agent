"""Google Vertex AI provider profile: Gemini via Google Cloud's OpenAI-compatible endpoint.

Auth is OAuth2 (service-account JSON or ADC), not a static key: ``agent/vertex_adapter.py``
mints ``(token, base_url)`` and the token is passed as ``api_key``. ``auth_type="vertex"``
keeps it out of the api_key provider path so a credentials-file path is never mistaken for a key.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


# Models that take the ``xhigh`` thinking level; Gemini 4 Argon and Carbon answer 400 to it.
_XHIGH_MODELS = ("gemini-4-hydrogen", "gemini-4-neon")
_XHIGH_EFFORTS = {"xhigh", "max", "ultra"}


def _with_top_thinking_level(model: str, reasoning_config: Any, thinking_config: dict[str, Any]) -> dict[str, Any]:
    """Send ``xhigh`` where the model has it: the shared mapping stops at ``high``, the top level of the
    other thinking-level models."""
    name = (model or "").strip().lower().removeprefix("google/")
    effort = str(reasoning_config.get("effort") or "").strip().lower() if isinstance(reasoning_config, dict) else ""
    if name.startswith(_XHIGH_MODELS) and effort in _XHIGH_EFFORTS and thinking_config.get("thinkingLevel") == "high":
        return {**thinking_config, "thinkingLevel": "xhigh"}
    return thinking_config


class VertexProfile(ProviderProfile):
    """Vertex AI — reuse Gemini's thinking_config translation for extra_body."""

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Emit ``extra_body.google.thinking_config`` like the ``gemini`` provider's OpenAI-compat subpath."""
        from agent.transports.chat_completions import _build_gemini_thinking_config, _snake_case_gemini_thinking_config

        model = context.get("model") or ""
        reasoning_config = context.get("reasoning_config")
        raw = _build_gemini_thinking_config(model, reasoning_config)
        if not raw:
            return {}
        raw = _with_top_thinking_level(model, reasoning_config, raw)
        thinking_config = _snake_case_gemini_thinking_config(raw)
        return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}


vertex = VertexProfile(
    name="vertex", aliases=("google-vertex", "vertex-ai", "gcp-vertex", "vertexai"), api_mode="chat_completions",
    env_vars=(),  # OAuth2 via service account / ADC — not a static key env var
    base_url="https://aiplatform.googleapis.com",  # real base_url computed at runtime
    auth_type="vertex", default_aux_model="google/gemini-3.6-flash",
    supports_model_listing=False,  # no /models route on the OpenAI-compat endpoint; setup ships a curated list
)

register_provider(vertex)
