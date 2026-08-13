"""Shared constants for cost and model-tier configuration.

Single source of truth for data/collect.py, api/main.py, models/baseline.py,
and eval/cost_quality.py so training, evaluation, and serving cost estimates
can't silently drift from each other.
"""

COST = {
    "cheap":     0.0,     # Groq free tier
    "expensive": 2.50,    # per 1M tokens, GPT-4o equivalent pricing benchmark
}

MODEL_NAMES = {
    "cheap":     "llama-3.1-8b-instant",
    "expensive": "llama-3.3-70b-versatile",
}