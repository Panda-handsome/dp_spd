"""Frozen-LLaMA TextObfuscator-style generation adapter.

The original TextObfuscator is a split RoBERTa classifier trained jointly
with task-specific prototype losses.  This package keeps its *split hidden
state + Laplace noise* operation available for an explicitly labelled,
non-equivalent WikiText-2/LLaMA comparison.
"""
