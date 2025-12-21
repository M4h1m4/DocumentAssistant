import os
from fastapi import FastAPI 
from dotenv import load_dotenv
from .api import router 
from .db_sql import init_db

load_dotenv() 
 
SQLITE_PATH: str= os.getenv("SQLITE_PATH", "./meta.db")


def create_app() -> FastAPI:
    app = FastAPI(title="PrecisBox", version="1.0.0")
    app.include_router(router)

    @app.on_event("startup")
    def _startup() -> None:
        init_db(SQLITE_PATH)
    
    return app

app = create_app()