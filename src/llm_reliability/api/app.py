"""
FastAPI application factory and lifecycle assembly.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llm_reliability.api.config import ApiSettings
from llm_reliability.api.dependencies import get_settings
from llm_reliability.api.errors import ApiException
from llm_reliability.api.middleware import RequestCorrelationMiddleware
from llm_reliability.api.routers import (
    analytics,
    benchmarks,
    diagnostics,
    experiments,
    health,
    providers,
    rag,
    robustness,
    runs,
)

logger = logging.getLogger("llm_reliability.api")


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Creates and configures a new FastAPI application instance."""
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: verify / create persistence root
        settings.persistence_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "FastAPI backend initialized",
            extra={
                "environment": settings.environment,
                "persistence_dir": str(settings.persistence_dir),
                "api_v1_prefix": settings.api_v1_prefix,
            },
        )
        yield
        # Shutdown
        logger.info("FastAPI backend shutting down")

    app = FastAPI(
        title="LLM Reliability & Evaluation Platform API",
        description=(
            "Versioned REST API for model evaluation, diagnostics, retrieval, and analytics."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url=f"{settings.api_v1_prefix}/docs",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    )

    # Route handlers resolve settings through the ``get_settings`` dependency.
    # Without this override an app built with explicit ``settings`` would still
    # read the process-wide defaults (for example the real ``./experiments``
    # store), so tests and embedded uses would silently write to the wrong place.
    app.dependency_overrides[get_settings] = lambda: settings

    # Middleware: Request correlation and timing
    app.add_middleware(RequestCorrelationMiddleware)

    # Middleware: CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Exception Handlers
    @app.exception_handler(ApiException)
    async def handle_api_exception(request: Request, exc: ApiException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {
                "loc": list(err.get("loc", [])),
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "The request payload or parameters failed validation.",
                    "details": errors,
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred.",
                    "details": None,
                }
            },
        )

    # Mount API v1 routers
    prefix = settings.api_v1_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(providers.router, prefix=prefix)
    app.include_router(benchmarks.router, prefix=prefix)
    app.include_router(experiments.router, prefix=prefix)
    app.include_router(runs.router, prefix=prefix)
    app.include_router(diagnostics.router, prefix=prefix)
    app.include_router(rag.router, prefix=prefix)
    app.include_router(robustness.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)

    return app


# Default application instance for Uvicorn
app = create_app()
