
from fastapi import APIRouter
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from typing import Optional, List, AnyStr
from json import JSONDecodeError
import re
import textstat

from util.utils import LangEnum, SpacySingleton

router = APIRouter()

@router.post("/chunked_sentence_complexity/",
             summary="Chunks a text and calculates complexity of it.",
             description=
"""
## Examples:
- Wow, this is really cool!
- fr
- la
""")
def chunked_sentence_complexity(text: Optional[str] = Form("Wow, this is really cool!"),
                                lang: Optional[LangEnum] = Form(LangEnum.EN)):

    complexity = "none"
    if text:
        textstat.set_lang(lang.name.lower())

        nlp = SpacySingleton.get_nlp(lang) # defaults to "en_core_web_sm"
        doc = nlp(text)

        complexities = [textstat.flesch_reading_ease(sent.text) for sent in doc.sents]

        avg = int(round(sum(complexities) / len(complexities)))
        complexity = get_mapping_complexity(avg)
    return {"overall_text_complexity": complexity}


@router.post("/maximum_sentence_complexity/",
             summary="Chunks a text and calculates complexity of it.",
             description=
"""
## Examples:
- An easy sentence. Despite the rains persistence, the resilient team continued their expedition, undeterred by the relentless downpour.
- fr
- la
""")
def maximum_sentence_complexity(text: Optional[str] = Form("An easy sentence. Despite the rains persistence, the resilient team continued their expedition, undeterred by the relentless downpour."),
                                lang: Optional[LangEnum] = Form(LangEnum.EN)):

    complexity = "none"
    if text:
        textstat.set_lang(lang.name.lower())
        nlp = SpacySingleton.get_nlp(lang) # defaults to "en_core_web_sm"
        doc = nlp(text)

        complexities = [textstat.flesch_reading_ease(sent.text) for sent in doc.sents]
        complexity = get_mapping_complexity(min(complexities))
    return {"overall_text_complexity": complexity}


def get_mapping_complexity(score):
    if score < 30:
        return "very difficult"
    if score < 50:
        return "difficult"
    if score < 60:
        return "fairly difficult"
    if score < 70:
        return "standard"
    if score < 80:
        return "fairly easy"
    if score < 90:
        return "easy"
    return "very easy"


@router.post("/difficult_words_extraction/",
             summary="Extracts difficult words in a given text",
             description=
             """
             ## Examples:
             - My cat is eleven years old. My Dad plays the saxophone. My brother mows the lawn with our lawnmower. The butterfly is colorful.
             - fr
             - la
             """)
def difficult_words_extraction(text: Optional[str] = Form("My cat is eleven years old. My Dad plays the saxophone. My brother mows the lawn with our lawnmower. The butterfly is colorful."),
                               lang: Optional[LangEnum] = Form(LangEnum.EN),
                               syllable_threshold: int = Form(3),
                               your_label: str = Form("difficult_word")):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    syllable_threshold = syllable_threshold
    difficult_words = textstat.difficult_words_list(text, syllable_threshold)

    pattern = "|".join(difficult_words)
    difficult_words = []
    for match in re.finditer(pattern, text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        difficult_words.append([your_label, span.start, span.end])

    return {f"{your_label}s": difficult_words}


@router.post("/syllable_count/",
             summary="Counts the number of sylabbles in a text.",
             description=
             """
             ## Examples:
             - This sentence has 7 syllables.
             - fr
             - la
             """)
def syllable_count(text: Optional[str] = Form("This sentence has 7 syllables.")):

    syllables = textstat.syllable_count(text)
    return {"syllableCount": syllables}


@router.post("/reading_time/",
             summary="Calculate the reading time of a text.",
             description=
             """
             ## Examples:
             - This sentence should take less than 1 second to read.
             - fr
             - la
             """)
def reading_time(text: Optional[str] = Form("This sentence should take less than 1 second to read."),
                 ms_per_char: float = Form(14.69)):

    time_to_read = textstat.reading_time(text, ms_per_char=ms_per_char)
    return {"readingTime": time_to_read}
