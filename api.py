from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from scalar_fastapi.scalar_fastapi import Layout

import nltk_api
import spacy_api
import textacy_api
import textblob_api
import textstat_api
from scalar_fastapi import get_scalar_api_reference

api = FastAPI()

origins = [
    "http://192.168.1.42:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://localhost",
    "http://localhost:4455",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",
    "https://bricks.kern.ai",
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/")
async def root():
    html_content = """
    <html>
        <head>
            <title>Kern AI - Bricks</title>
        </head>
        <body>
            <h1>Endpoints for <a href="https://kern.ai" target="_blank">Kern AI</a> bricks</h1>
            <p>Please look into bricks to see how to use these endpoints.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@api.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=api.openapi_url,
        title=api.title,
        layout=Layout.MODERN
    )


# download_all_models()

api.include_router(spacy_api.router, prefix="/spacy", tags=["spacy"])
api.include_router(textstat_api.router, prefix="/textstat", tags=["textstat"])
api.include_router(nltk_api.router, prefix="/nltk", tags=["nltk"])
api.include_router(textacy_api.router, prefix="/textacy", tags=["textacy"])
api.include_router(textblob_api.router, prefix="/textblob", tags=["textblob"])
