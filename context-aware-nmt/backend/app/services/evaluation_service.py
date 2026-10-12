"""Service layer stubs for evaluation orchestration.

Full implementation will:
  * Seed PRNGs for determinism.
  * Run both pipelines over a test set.
  * Compute every requested metric through the ``evaluation`` registry.
  * Persist aggregate + per-sentence arrays to ``evaluation_results``.
"""
