"""Plain-English claim explanations via Groq, with template fallback."""
from __future__ import annotations

import os
from typing import Any


def _template_explanation(
    probability: float,
    band: str,
    contributions: list[dict[str, Any]],
) -> str:
    top = contributions[:3]
    parts = []
    for c in top:
        direction = "raises" if float(c.get("shap", 0)) > 0 else "lowers"
        parts.append(f"{c.get('feature')}={c.get('value')} ({direction} risk)")
    drivers = "; ".join(parts) if parts else "available claim features"
    return (
        f"This claim is scored as {band} risk with an estimated fraud probability "
        f"of {probability:.1%}. The strongest drivers are: {drivers}. "
        f"Treat this as a screening signal for an adjuster review, not an automated decision."
    )


def explain_claim(
    probability: float,
    band: str,
    contributions: list[dict[str, Any]],
    claim: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Returns {"explanation": str, "source": "groq"|"template"}.
    Uses GROQ_API_KEY from the environment when present.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        return {
            "explanation": _template_explanation(probability, band, contributions),
            "source": "template",
        }

    try:
        from openai import OpenAI

        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = OpenAI(api_key=api_key, base_url=base_url)

        contrib_lines = "\n".join(
            f"- {c.get('feature')}: value={c.get('value')}, shap={float(c.get('shap', 0)):.4f}"
            for c in contributions[:6]
        )
        prompt = (
            "You are an insurance SIU assistant. Write 2-3 short plain-English sentences "
            "explaining why this vehicle insurance claim received its fraud risk score. "
            "Do not accuse the claimant. Emphasize that this is a screening aid. "
            f"Probability={probability:.3f}, band={band}.\n"
            f"Top SHAP contributions (positive pushes toward fraud):\n{contrib_lines}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Explain model scores briefly for claims adjusters.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty completion")
        return {"explanation": text, "source": "groq"}
    except Exception:
        return {
            "explanation": _template_explanation(probability, band, contributions),
            "source": "template",
        }
