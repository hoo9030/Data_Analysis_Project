import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

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
        # Redirect root to the web UI
        return RedirectResponse(url="/web", status_code=307)

    app.include_router(api_router, prefix="/api")

    # Mount simple web UI for showcasing features
    static_dir = Path(__file__).resolve().parent / "web"
    if static_dir.exists():
        app.mount("/web", StaticFiles(directory=str(static_dir), html=True), name="web")

        @app.get("/ui")
        def ui_index():
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "UI not found"}

    return app


app = create_app()
