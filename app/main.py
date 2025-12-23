"""
RAG Playground Backend - Main Application Entry Point
"""

import logging
import sys

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import documents, rag, results
from app.core.config import get_settings
from app.core.health import check_ollama_health, get_ollama_setup_instructions
from app.db.database import create_db_and_tables

# Configure logging
settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Playground Backend",
    version="0.1.0",
    description="Local-first RAG experimentation platform",
)

# Configure CORS - Allow localhost on any port (for development flexibility)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|\[::1\]):\d+",  # Allow any localhost port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(documents.router)
app.include_router(rag.router)
app.include_router(results.router)


@app.on_event("startup")
async def startup_health_check():
    """Initialize database and verify Ollama is available at startup"""
    # Create database tables
    create_db_and_tables()
    logger.info("Database tables created/verified")

    # Verify Ollama is available
    is_healthy, message = await check_ollama_health()
    if not is_healthy:
        logger.error(message)
        logger.error(get_ollama_setup_instructions())
        sys.exit(1)
    logger.info("Ollama health check passed")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG Playground Backend"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    is_healthy, message = await check_ollama_health()
    return {"healthy": is_healthy, "message": message}
