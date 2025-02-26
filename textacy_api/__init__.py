import re
from typing import Optional

import textacy
from fastapi import APIRouter
from fastapi import Form

from util.utils import LangEnum, SpacySingleton

router = APIRouter()

@router.post("/verb_phrase_extraction/",
             summary="Extracts the verb phrases from a record.",
             description=
             """
             ## Examples:
             - In the next section, we will build a new model which is more accurate than the previous one.
             - fr
             - la
             
             ## TODO:
             - See if it's working in Latin
             """)
def verb_phrase_extraction(text: Optional[str] = Form("In the next section, we will build a new model which is more accurate than the previous one."),
                           lang: Optional[LangEnum] = Form(LangEnum.EN)):

    patterns = [{"POS": "AUX"}, {"POS": "VERB"}]
    doc = textacy.make_spacy_doc(text, lang=lang.value)
    verb_phrase = textacy.extract.token_matches(doc, patterns=patterns)
    verb_chunk = []
    for chunk in verb_phrase:
        verb_chunk.append(["match", chunk.start, chunk.end])

    return {"action": verb_chunk}