"""Temporary internal identifiers. The real address is hashed with a random per-session key and never stored."""
from __future__ import annotations

import hashlib
import hmac
import os


class IdMapper:
    def __init__(self, secret: bytes | None = None):
        self._secret = secret or os.urandom(16)

    def key(self, address: str) -> str:
        h = hmac.new(self._secret, address.upper().encode(), hashlib.sha256).hexdigest()
        return "D-" + h[:8]

    def rotate(self) -> None:
        """Start a new identifier space. Older identifiers can no longer be matched to addresses."""
        self._secret = os.urandom(16)
