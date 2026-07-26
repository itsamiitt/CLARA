"""Entrypoint for running CLARA as an API service.

Binds to localhost by default: the memory store holds everything the agent has
ever learned, and the API's own auth is opt-in (``CLARA_AUTH_REQUIRED`` plus
``CLARA_API_TOKENS``). Set ``CLARA_API_HOST`` deliberately — and only with auth
configured or an authenticating gateway in front — to listen more widely.
"""

import os

from clara.api.app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("CLARA_API_HOST", "").strip() or "127.0.0.1"
    port = int(os.environ.get("CLARA_API_PORT", "").strip() or 8000)
    # reload= is a development convenience that also re-executes code on file
    # change; it stays opt-in rather than being the shipped default.
    reload = os.environ.get("CLARA_API_RELOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    uvicorn.run("clara.main:app", host=host, port=port, reload=reload)
