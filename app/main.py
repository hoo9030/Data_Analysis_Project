import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import api_router


def create_app() -> FastAPI:
    app = FastAPI(title="Data Analysis Studio (Clean)", version=__version__)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "name": "Data Analysis Studio",
            "message": "Fresh start",
            "version": __version__,
        }

    app.include_router(api_router, prefix="/api")

    return app


app = create_app()

