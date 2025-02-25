from fastapi import APIRouter
from spacy.spacy import SpacySingleton, LangEnum
from pydantic import BaseModel, Field
import re

router = APIRouter()

class TextLang(BaseModel):
    text: str
    lang: LangEnum | None = LangEnum.AUTO

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "I live at 35 Wood Lane, Pilsbury ME19 7DY, United Kingdom. But I have also lived at 221BE Baker-callum Street, London VIC 3SX, United Kingdom.",
                    "lang": LangEnum.EN.name
                },
                {
                    "text": "Francais.",
                    "lang": LangEnum.FR.name
                },
                {
                    "text": "Latin.",
                    "lang": LangEnum.LA.name
                },
            ]
        }
    }

@router.post("/spacy/address_extraction/")
def address_extraction(text_lang: TextLang, lang: LangEnum | None = LangEnum.AUTO):

    nlp = SpacySingleton.get_nlp(lang)

    text = text_lang.text
    doc = nlp(text)

    regex_1 = re.compile(
        r"(?:\d{1,5}(?:[A-Z ]+[ ]?)+(?:[A-Za-z-]+[ ]?)+(?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr(?:\.)?|Rd(?:\.)?|Blvd(?:\.)?|Ln(?:\.)?|St(?:\.)?|Strasse|Hill|Alley|Alle|City)[,](?:[ A-Za-z0-9,]+[ ]?)?)"
    )
    regex_2 = re.compile(
        r"(?:(?:[A-Za-z-]?)+[ ](?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr(?:\.)?|Rd(?:\.)?|Blvd(?:\.)?|Ln(?:\.)?|St(?:\.)?|Strasse|Str(?:\.)?|Hill|Alley|Alle|City)[ ]+\d{1,5},(?:[ A-Za-z0-9,]+[ ]?)?)"
    )
    addresses = []

    if regex_1.findall(text):
        for match in regex_1.finditer(text):
            start, end = match.span()
            print(start, end)
            span = doc.char_span(start, end, alignment_mode="expand")
            print(span)
            addresses.append(["address", span.start, span.end, span.text])
    if regex_2.findall(text):
        for match in regex_2.finditer(text):
            start, end = match.span()
            span = doc.char_span(start, end, alignment_mode="expand")
            print(span)
            addresses.append(["address", span.start, span.end])

    return {"addresses": addresses}
