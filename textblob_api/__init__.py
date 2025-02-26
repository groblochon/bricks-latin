
from fastapi import APIRouter
from nltk.corpus import stopwords
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from typing import Optional, List, AnyStr
from json import JSONDecodeError
import re
from textblob import TextBlob

from util.utils import LangEnum, SpacySingleton

router = APIRouter()

@router.post("/aspect_extraction/",
             summary="Matches aspects in a text to positive or negative sentiment.",
             description=
"""
## Examples:
- It has a really great battery life, but I hate the window size...
- fr
- la
""")
def aspect_extraction(text: Optional[str] = Form("It has a really great battery life, but I hate the window size..."),
                      lang: Optional[LangEnum] = Form(LangEnum.EN),
                      windows_size: int = Form(4),
                      sensitivity: float = Form(0.5)):

    matches = []
    if text:
        nlp = SpacySingleton.get_nlp(lang)
        doc = nlp(text)

        for chunk in doc.noun_chunks:
            left_bound = max(chunk.sent.start, chunk.start - (windows_size // 2) + 1)
            right_bound = min(chunk.sent.end, chunk.end + (windows_size // 2) + 1)
            window_doc = doc[left_bound:right_bound]
            sentiment = TextBlob(window_doc.text).polarity
            if sentiment < -(1 - sensitivity):
                matches.append(["negative", chunk.start, chunk.end])
            elif sentiment > (1 - sensitivity):
                matches.append(["positive", chunk.start, chunk.end])

    return {"aspects": matches}
