from typing import Optional
from fastapi import APIRouter
from fastapi import Form
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

router = APIRouter()

@router.post("/sumy_website_summarizer/",
             summary="Summarize a website using sumy.",
             description=
             """
             ## Examples:
             - https://en.wikipedia.org/wiki/capybara
             - fr
             - la
             
             ## TODO:
             - see for Latin
             """)
def sumy_website_summarizer(url: Optional[str] = Form('https://en.wikipedia.org/wiki/capybara'),
                            language: str = Form('english'),
                            sentence_count: int = Form(5)):

    parser = HtmlParser.from_url(url, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])
