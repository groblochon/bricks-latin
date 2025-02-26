import html
import unicodedata
from typing import Optional
from urllib.parse import urlsplit

from LeXmo import LeXmo
from better_profanity import profanity
from bs4 import BeautifulSoup
from fastapi import APIRouter
from fastapi import Form
from translate import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from util.utils import LangEnum

import dateutil.parser as dparser
import holidays
from langdetect import detect

router = APIRouter()

@router.post("/language_detection/",
             summary="Detects the language of a given text.",
             description=
             """
             ## Examples:
             - This is an english sentence.
             - fr
             - la
             """)
def language_detection(text: str = Form("This is an english sentence.")):

    if not text or not text.strip():
        return {"language": "unknown"}
    return {"language": detect(text)}


@router.post("/newline_splitter/",
             summary="Splits a text by newline characters.",
             description=
             """
             ## Examples:
             - This is the first line.
    And this is the second line.
    Here's a third one, too.
             - fr
             - la
             """)
def newline_splitter(text: Optional[str] = Form("""This is the first line.
    And this is the second line.
    Here's a third one, too.
    """)):

    splits = [t.strip() for t in text.split("\n")]
    return {"splitted_text" : [val for val in splits if len(val) > 0]}

@router.post("/word_count_classifier/",
             summary="Checks the length of a string by counting the number of words in it",
             description=
             """
             ## Examples:
             - This is too short!
             - fr
             - la
             """)
def word_count_classifier(text: Optional[str] = Form('This is too short!')):

    """Checks the length of a string by counting the number of words in it"""
    words = text.split()
    length = len(words)
    if length < 5:
        return {"text_length": "short"}
    elif length < 20:
        return {"text_length": "medium"}
    else:
        return {"text_length": "long"}


@router.post("/special_character_classifier/",
             summary="Checks if a string contains special characters",
             description=
             """
             ## Examples:
             - Super funny haha 😀.
             - fr
             - la
             """)
def special_character_classifier(text: Optional[str] = Form('Super funny haha 😀.')):

    allowed_range = ALLOWED_RANGE

    for char in text:
        if ord(char) not in allowed_range and unicodedata.category(char) != "Zs":
            return {"contains_special_char": True}
    return {"contains_special_char": False}

ALLOWED_RANGE = set(range(32, 127)).union( # Basic Latin
    set(range(160, 255)), # Latin-1 Supplement
    set(range(256, 384)),  # Latin Extended-A
    set(range(384, 592)),  # Latin Extended-B
    set(range(8192, 8303)),  # General Punctuation
    set(range(8352, 8399)),  # Currency Symbols
    set([ord("\t"), ord("\n"), ord("\r")])# common stop chars
)


@router.post("/vader_sentiment_classifier/",
             summary="Get the sentiment of a text using the VADER algorithm.",
             description=
             """
             ## Examples:
             - World peace announced by the United Nations.
             - fr
             - la
             """)
def vader_sentiment_classifier(text: Optional[str] = Form('World peace announced by the United Nations.')):

    analyzer = SentimentIntensityAnalyzer()

    vs = analyzer.polarity_scores(text)
    if vs["compound"] >= 0.05:
        return {"sentiment": "positive"}
    elif vs["compound"] > -0.05:
        return {"sentiment": "neutral"}
    elif vs["compound"] <= -0.05:
        return {"sentiment": "negative"}


@router.post("/profanity_detection/",
             summary="Detects if a given text contains abusive language.",
             description=
             """
             ## Examples:
             - You suck man!
             - fr
             - la
             """)
def profanity_detection(text: str = Form('You suck man!')):

    result = profanity.contains_profanity(text)
    return {"profanity": result}


@router.post("/emotionality_detection/",
             summary="Fetches emotions from a given text",
             description=
             """
             ## Examples:
             - As Harry went inside the Chamber of Secrets, he discovered the Basilisk's layer. Before him stood Tom
            Riddle, with his wand. Harry was numb for a second as if he had seen a ghost. Moments later the giant 
            snake attacked Harry but fortunately, Harry dodged and ran into one of the sewer lines while the serpent 
            followed. The Basilisk couldn't be killed with bare hands but only with a worthy weapon.
             - fr
             - la
             """)
def emotionality_detection(text: str = Form("""As Harry went inside the Chamber of Secrets, he discovered the Basilisk's layer. Before him stood Tom
            Riddle, with his wand. Harry was numb for a second as if he had seen a ghost. Moments later the giant 
            snake attacked Harry but fortunately, Harry dodged and ran into one of the sewer lines while the serpent 
            followed. The Basilisk couldn't be killed with bare hands but only with a worthy weapon.""")):

    try:
        emo = LeXmo.LeXmo(text)
        del emo["text"]
        del emo["positive"]
        del emo["negative"]
        unique = dict(zip(emo.values(), emo.keys()))
        if len(unique) == 1:
            return "Cannot determine emotion"
        else:
            emo = max(emo, key=emo.get)
            return {"emotion": emo}
    except ValueError:
        return "Valid text required"


@router.post("/workday_classifier/",
             summary="Checks if a date is a workday, weekend or a holiday.",
             description=
             """
             ## Examples:
             - 01.01.2023 is a holiday in Germany.
             - fr
             - la
             """)
def workday_classifier(text: Optional[str] = Form('01.01.2023 is a holiday in Germany.'),
                      lang: Optional[LangEnum] = Form(LangEnum.EN)):

    # try to parse the date from the string
    try:
        date = dparser.parse(text, fuzzy=True).date()
    except:
        return "No date found or invalid date"

    # check if country code is specified
    country_code = lang.name
    if country_code:
        national_holidays = holidays.country_holidays(f"{country_code}")
        if date in national_holidays:
            return {"weekdayType": "Holiday"}

    # check if weekday is a workday or a weekend
    if date.weekday() < 5:
        return {"weekdayType": "Working day"}
    else:
        return {"weekdayType": "Weekend"}


@router.post("/language_translator/",
             summary="Function to translate text.",
             description=
             """
             ## Examples:
             - Salut, comment allez-vous ?.
             - fr
             - la
             """)
def language_translator(text: Optional[str] = Form('Salut, comment allez-vous ?'),
                        lang: Optional[LangEnum] = Form(LangEnum.FR),
                        lang_to: Optional[LangEnum] = Form(LangEnum.EN)):

    translator = Translator(from_lang=lang.name.lower(), to_lang=lang_to.name.lower())
    translation = translator.translate(text)
    return {"translation": translation}


@router.post("/spelling_check/",
             summary="Parses a domain of a URL.",
             description=
             """
             ## Examples:
             - https://huggingface.co/sentence-transformers
             - fr
             - la
             """)
def spelling_check(url: str = Form("https://huggingface.co/sentence-transformers")):

    if "http" in url:
        parser = urlsplit(url)
        domain = parser.netloc
    else:
        part = url.strip('/').split('/')
        domain = part[0]
    if "www." in domain:
        domain = domain.lstrip("www.")
    return domain


@router.post("/html_unescape/",
             summary="Function to translate text.",
             description=
             """
             ## Examples:
             - Here&#39;s how &quot;Kern.ai Newsletter&quot; did today. 3. "World&#8217;s largest tech conference: &quot;Innovate 2023&#8482;&quot; begins tomorrow!
             - fr
             - la
             """)
def html_unescape(text: Optional[str] = Form('Here&#39;s how &quot;Kern.ai Newsletter&quot; did today. 3. "World&#8217;s largest tech conference: &quot;Innovate 2023&#8482;&quot; begins tomorrow!')):

    unescaped_text = html.unescape(text)

    return {"Unescaped text": unescaped_text}


@router.post("/html_cleanser/",
             summary="Removes the HTML tags from a text.",
             description=
             """
             ## Examples:
             - <!DOCTYPE html>
                <html>
                <body>
                <h1>Website header</h1>
                <p>
                Hello world.
                My website is live!
                </p>
                </body>
                </html>
             - fr
             - la
             """)
def html_cleanser(text: Optional[str] = Form("""
            <!DOCTYPE html>
            <html>
            <body>
            <h1>Website header</h1>
            <p>
            Hello world.
            My website is live!
            </p>
            </body>
            </html>
            """)):

    soup = BeautifulSoup(text, "html.parser")

    # Remove any line breakers as well
    text = soup.text.splitlines()
    text = " ".join([w for w in text if len(w) >= 1])

    return {"Cleaned text": text}