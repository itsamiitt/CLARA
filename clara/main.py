"""Entrypoint for running CLARA as an API service."""

from clara.api.app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("clara.main:app", host="0.0.0.0", port=8000, reload=True)
