"""Vercel Serverless Function entrypoint exporting the FastAPI application."""

from app.main import app

__all__ = ["app"]
