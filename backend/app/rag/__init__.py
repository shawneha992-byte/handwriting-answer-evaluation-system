"""Retrieval-augmented grading of descriptive answers.

Pipeline: teacher uploads reference material -> ingest (extract, chunk, embed, store)
-> for each descriptive answer retrieve the relevant chunks -> a text LLM judges the
answer against the rubric and the retrieved evidence -> plain code validates, clamps
and totals the marks. The LLM judges; code decides what is allowed to become a mark.
"""
