"""Synchronous SDK transport with cancellable deadlines through the entire body."""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar

import httpx2

# Set by the synchronous request owner; capacity waits share the same deadline.
provider_request_deadline: ContextVar[float | None] = ContextVar("provider_request_deadline", default=None)


class DeadlineTransport(httpx2.BaseTransport):
    """Run owned asynchronous HTTP I/O to completion or cancellation before returning."""

    def __init__(self) -> None:
        self._runner = asyncio.Runner()
        self._client = httpx2.AsyncClient(follow_redirects=False)
        self._closed = False

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        """Bound headers and all body reads by one monotonic deadline."""

        deadline = provider_request_deadline.get()
        explicit = request.extensions.get("provider_deadline")
        if isinstance(explicit, (float, int)):
            deadline = float(explicit) if deadline is None else min(deadline, explicit)
        timeouts = request.extensions.get("timeout", {})
        values = [value for value in timeouts.values() if isinstance(value, (float, int))]
        if values:
            attempt_deadline = time.monotonic() + min(values)
            deadline = attempt_deadline if deadline is None else min(deadline, attempt_deadline)
        # Pass the absolute value explicitly: Runner retains its initial Context.
        return self._runner.run(self._send(request, deadline))

    async def _send(self, request: httpx2.Request, deadline: float | None) -> httpx2.Response:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        asynchronous = httpx2.Request(
            request.method,
            request.url,
            headers=request.headers,
            content=request.content,
            extensions=request.extensions,
        )
        try:
            async with asyncio.timeout(remaining):
                response = await self._client.send(asynchronous, stream=True)
                try:
                    # Retain the wire encoding: the synchronous response must
                    # decompress exactly once, including for gzip responses.
                    content = b"".join([chunk async for chunk in response.aiter_raw()])
                finally:
                    await response.aclose()
        except TimeoutError:
            raise httpx2.ReadTimeout("Provider absolute request deadline exceeded.", request=request) from None
        return httpx2.Response(
            response.status_code,
            headers=response.headers,
            content=content,
            extensions={key: value for key, value in response.extensions.items() if key != "network_stream"},
        )

    def close(self) -> None:
        """Close every owned connection and the event loop, including after cancellation."""

        if self._closed:
            return
        self._closed = True
        try:
            self._runner.run(self._client.aclose())
        finally:
            self._runner.close()
