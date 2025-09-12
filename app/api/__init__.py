from fastapi import APIRouter

from .health import router as health_router
from .datasets import router as datasets_router

api_router = APIRouter()

# Basic health and version endpoints
api_router.include_router(health_router, tags=["health"])

# Dataset management endpoints
api_router.include_router(datasets_router)
