from fastapi import APIRouter

from .. import __version__


router = APIRouter()


@router.get("/info")
def info():
    return {
        "name": "Data Analysis Studio",
        "message": "Fresh start",
        "version": __version__,
    }

