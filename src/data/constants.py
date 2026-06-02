"""Shared label/span constants for training targets.

Kept outside ``collator.py`` so objectives and backbone hooks can import these
values without creating a circular import through ``src.models``.
"""

LABEL_IGNORE = -100  # HF's "ignore this position in the loss" sentinel

# Span codes for explanation-aware loss weighting. Each TARGET token is tagged
# so objectives can distinguish answer tokens from explanation tokens.
SPAN_IGNORE = 0
SPAN_EXPLANATION = 1
SPAN_ANSWER = 2

SPAN_CODE = {"explanation": SPAN_EXPLANATION, "answer": SPAN_ANSWER}
