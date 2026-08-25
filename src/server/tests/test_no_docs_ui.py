"""The server serves a schema, not a documentation UI.

FastAPI's `/docs` and `/redoc` load Swagger UI and ReDoc from a CDN, and this
server sends `script-src 'self'` — so those pages are refused by the browser and
render blank while still answering 200. A broken page that reports success is
worse than no page, and the fixes both cost more than the feature: vendoring the
bundles means carrying megabytes of third-party browser code through the build,
and a CSP exception means weakening the header for every document.

So both are off, and `/openapi.json` is the documentation surface. These tests
hold that decision: a future `FastAPI(...)` that forgets `docs_url=None` brings
the blank page back silently, and nothing else in the suite would notice.
"""

from __future__ import annotations

from qds.app import create_app, create_recovery_app

from .conftest import make_client


def test_no_documentation_pages(client):
    """404, not a page that renders empty."""
    for page in ("/docs", "/redoc"):
        assert client.get(page).status_code == 404, f"{page} is served again"


def test_openapi_schema_is_served(client):
    """The schema is the surface clients actually read, and must stay."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    body = response.json()
    assert body["info"]["title"] == "Quantum Diffusion Server"
    # Not merely valid JSON: the endpoints have to be described in it, or the
    # schema is being served while documenting nothing.
    assert "/v1/images/generations" in body["paths"]


def test_app_declares_no_docs_urls(settings, engine):
    """The app object itself, so the assertion does not depend on routing."""
    app = create_app(settings, engine)

    assert app.docs_url is None
    assert app.redoc_url is None


def test_recovery_app_serves_no_docs_either(settings):
    """The recovery app is a way in when configuration is broken.

    It gets the same treatment as the main app: a second FastAPI construction
    is exactly where a default quietly comes back.
    """
    app = create_recovery_app(settings, "test")

    assert app.docs_url is None
    assert app.redoc_url is None

    with make_client(app) as client:
        assert client.get("/docs").status_code == 404
