"""
Shared header-phrase pools for the "labeled-sections" trace-presentation
style used across the SYNCR training tasks (CLEVRER + Habitat).

Each task's trace builders already pick between a few presentation styles at
random per example (labeled sections vs. flowing prose vs. compact evidence
blocks -- see the *_VARIANTS lists in clvr_temporal_alignment.py and
habitat_traces.py). But the "labeled sections" style used the exact same
literal header strings -- "**Grounding:**", "**Observation:**",
"**Deduction:**", "**Conclusion:**" -- verbatim, for every task and every
example that happened to pick it. That is the single string most likely to
have leaked onto MVU-Eval outputs as a copied surface-form habit: the
fine-tuned model was directly observed emitting "**Grounding:**" on
unrelated, non-SYNCR questions. Drawing the header word from a small synonym
pool per example keeps the same labeled-sections structure (still a useful
style for some fraction of examples) but removes the one repeated literal
token that let the model key on "this exact string means apply reasoning
mode" regardless of content.
"""

import random

GROUNDING_HEADERS = [
    "**Grounding:**",
    "**Setup:**",
    "**Approach:**",
    "**Context:**",
    "**Framing:**",
]

OBSERVATION_HEADERS = [
    "**Observation:**",
    "**Evidence:**",
    "**What we see:**",
    "**Findings:**",
]

DEDUCTION_HEADERS = [
    "**Deduction:**",
    "**Reasoning:**",
    "**Analysis:**",
    "**Inference:**",
]

CONCLUSION_HEADERS = [
    "**Conclusion:**",
    "**Result:**",
    "**Answer:**",
    "**Therefore:**",
]


def header(pool):
    return random.choice(pool)
