"""Structured JSON logging with a per-request trace id."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get(),
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("fraud")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return root.getChild(name.replace("app.", ""))
