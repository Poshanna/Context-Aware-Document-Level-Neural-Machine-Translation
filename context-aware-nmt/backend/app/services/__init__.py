"""Service layer — business logic orchestration lives here.

RULE (from project_rules.md §3.1 / §3.2):
  Services MUST NOT import FastAPI ``Request``/``Response`` objects, and
  they MUST be callable from plain Python (tests, scripts, CLI) without
  a web server running.
"""
