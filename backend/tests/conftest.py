"""Existing exam tests call the API without logging in, so the login gate is switched off for them.
tests/test_auth.py switches it back on for its own checks."""
import os

os.environ.setdefault("AUTH_REQUIRED", "false")
