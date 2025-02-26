
from fastapi import APIRouter
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from typing import Optional, List, AnyStr
from json import JSONDecodeError
import re

from util.utils import LangEnum, SpacySingleton

router = APIRouter()

@router.post("/smalltalk_extraction/",
             summary="Detects smalltalk languages from chats",
             description=
             """
             ## Examples:
             - "Hello, how are you?" he asked.
               "I am doing fine, and you?", she said.
               "I am doing good as well.".
               "Listen, I wanted to talk to you about the something. Actually your car broke down on the bridge and I 
               suspect that the engine is heated up.".
               "Don't worry about that, I'll buy a new car!"
             - fr
             - la
             
             ## TODO:
             - add latin stopwords
             """)
def smalltalk_extraction(text: Optional[str] = Form('''"Hello, how are you?" he asked.
                                                        "I am doing fine, and you?", she said.
                                                        "I am doing good as well.".
                                                        "Listen, I wanted to talk to you about the something. Actually your car broke down on the bridge and I 
                                                        suspect that the engine is heated up.".
                                                        "Don't worry about that, I'll buy a new car!"'''),
                         lang: Optional[LangEnum] = Form(LangEnum.EN),
                         stop_words: str = Form("english")):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    sw = stopwords.words(stop_words)
    regex = re.compile(r"\".*?\"")

    smalltalk = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        text_list_original = span.text.replace('"', '').replace(',', '').split()
        new_text = []
        stop_words = []
        for token in text_list_original:
            if token not in sw:
                new_text.append(token)
            else:
                stop_words.append(token)
        if len(new_text) < 0.5 * len(text_list_original) or len(stop_words) < 8:
            smalltalk.append(["smalltalk", span.start, span.end])

    return {"smalltalk": smalltalk}


@router.post("/synonym_extraction/",
             summary="Detects smalltalk languages from chats",
             description=
             """
             ## Examples:
             - My sister is good at playing football.
             - fr
             - la
             
             ## TODO:
             - See if it's working in Latin
             """)
def synonym_extraction(text: Optional[str] = Form("My sister is good at playing football."),
                       lang: Optional[LangEnum] = Form(LangEnum.EN),
                       target_word: str = Form("Soccer")):

    # Find synonyms using wordnet
    synonyms = []
    for syn in wordnet.synsets(target_word):
        for i in syn.lemmas():
            synonyms.append(i.name())

    # Word are sometimes connected by a _, which we want to remove
    split_synonyms = [item.split(sep="_") for item in synonyms]

    # Break up potential list of lists into a single list
    combined_synonyms = [item for sublist in split_synonyms for item in sublist]

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    # Get the span of found matches
    synonym_matches = []
    for word in combined_synonyms:
        try:
            pattern = rf"({word})"
            match = re.search(pattern, text)

            start, end = match.span()
            span = doc.char_span(start, end, alignment_mode="expand")

            synonym_matches.append(["synonym", span.start, span.end])
        except:
            pass

    return {"synonyms": synonym_matches}


@router.post("/nltk_ngram_generator/",
             summary="Generate word n-grams from the input sentence. Punctuation and stop words are preserved.",
             description=
             """
             ## Examples:
             - Despite the unpredictable weather, the enthusiastic crowd gathered at the park for the annual summer festival, eagerly anticipating an evening filled with music, food, and vibrant celebrations.
             - fr
             - la
             
             ## TODO:
             - See if it's working in Latin
             """)
def nltk_ngram_generator(text: Optional[str] = Form("Despite the unpredictable weather, the enthusiastic crowd gathered at the park for the annual summer festival, eagerly anticipating an evening filled with music, food, and vibrant celebrations."),
                         lang: Optional[LangEnum] = Form(LangEnum.EN),
                         ngram_size: int = Form(2)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    tokens = [token.text for token in doc]
    n_grams = list(ngrams(tokens, ngram_size))

    return {"n_grams": n_grams}