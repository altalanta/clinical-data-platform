from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .phi_redaction import read_only_enabled

BLOCKED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class ReadOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if read_only_enabled() and request.method in BLOCKED_METHODS:
            return JSONResponse(
                {"detail": "Read-only mode: write methods are disabled."}, status_code=403
            )
        return await call_next(request)
