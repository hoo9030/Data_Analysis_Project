from fastapi import APIRouter

from .health import router as health_router

api_router = APIRouter()

# Basic health and version endpoints
api_router.include_router(health_router, tags=["health"])

