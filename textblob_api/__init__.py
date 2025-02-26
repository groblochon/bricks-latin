
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

@router.post("/textblob_spelling_correction/",
             summary="Correct spelling mistakes in a text using the TextBlob library.",
             description=
             """
             ## Examples:
             - His text contaisn some speling errors.
             - fr
             - la
             """)
def textblob_spelling_correction(text: str = Form("His text contaisn some speling errors.")):

    textblob_text = TextBlob(text)
    return {"correctedText": str(textblob_text.correct())}


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


@router.post("/textblob_sentiment/",
             summary="Calculate sentiment of a text.",
             description=
             """
             ## Examples:
             - Wow, this is awesome!
             - fr
             - la
             """)
def textblob_sentiment(text: Optional[str] = Form("Wow, this is awesome!"),
                       lang: Optional[LangEnum] = Form(LangEnum.EN)):

    blob = TextBlob(text)

    return {"sentiment": get_mapping_sentiment(blob.sentiment.polarity * 100)}

def setall(d, keys, value):
    for k in keys:
        d[k] = value


SENTIMENT_MAX_SCORE = 100
SENTIMENT_MIN_SCORE = -100

SENTIMENT_OUTCOMES = {}
setall(SENTIMENT_OUTCOMES, range(40, SENTIMENT_MAX_SCORE + 1), "very positive")
setall(SENTIMENT_OUTCOMES, range(20, 40), "positive")
setall(SENTIMENT_OUTCOMES, range(-20, 20), "neutral")
setall(SENTIMENT_OUTCOMES, range(-40, -20), "negative")
setall(SENTIMENT_OUTCOMES, range(SENTIMENT_MIN_SCORE, -40), "very negative")


def get_mapping_sentiment(score):
    if score < SENTIMENT_MIN_SCORE:
        return SENTIMENT_OUTCOMES[SENTIMENT_MIN_SCORE]
    return SENTIMENT_OUTCOMES[int(score)]


@router.post("/textblob_subjectivity/",
             summary="Calculate subjectivity of a text.",
             description=
             """
             ## Examples:
             - Wow, this is awesome!
             - fr
             - la
             """)
def textblob_subjectivity(text: str = Form("Wow, this is awesome!")):

    blob = TextBlob(text)

    return {"subjectivity": get_mapping_subjectivity(blob.sentiment.subjectivity * 100)}


SUBJECTIVITY_MAX_SCORE = 100
SUBJECTIVITY_MIN_SCORE = 0

SUBJECTIVITY_OUTCOMES = {}
setall(SUBJECTIVITY_OUTCOMES, range(80, SUBJECTIVITY_MAX_SCORE + 1), "subjective")
setall(SUBJECTIVITY_OUTCOMES, range(60, 80), "rather subjective")
setall(SUBJECTIVITY_OUTCOMES, range(40, 60), "neutral")
setall(SUBJECTIVITY_OUTCOMES, range(20, 40), "rather objective")
setall(SUBJECTIVITY_OUTCOMES, range(SUBJECTIVITY_MIN_SCORE, 20), "objective")


def get_mapping_subjectivity(score):
    if score < SUBJECTIVITY_MIN_SCORE:
        return SUBJECTIVITY_OUTCOMES[SUBJECTIVITY_MIN_SCORE]
    return SUBJECTIVITY_OUTCOMES[int(score)]

