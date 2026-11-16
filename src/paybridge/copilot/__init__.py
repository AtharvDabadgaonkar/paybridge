"""Grounded investigation copilot.

Pipeline: question -> intent detection -> deterministic Python analytics ->
allowlisted evidence pack -> LLM explanation (optional) -> claim validation ->
final response. The LLM never computes a number and never decides anything.
"""
