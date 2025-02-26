
import re
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter
from fastapi import Form
from nltk import ngrams
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import words

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


@router.post("/smalltalk_truncation/",
             summary="Removes all the irrelevant text from a passage or chats",
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
def smalltalk_truncation(text: Optional[str] = Form('''"Hello, how are you?" he asked.
                                                        "I am doing fine, and you?", she said.
                                                        "I am doing good as well.".
                                                        "Listen, I wanted to talk to you about the something. Actually your car broke down on the bridge and I 
                                                        suspect that the engine is heated up.".
                                                        "Don't worry about that, I'll buy a new car!"'''),
                         stop_words: str = Form("english")):

    sw = stopwords.words(stop_words)
    regex = re.compile(r"\".*?\"")

    remove_smalltalk = []
    for message in regex.findall(text):
        chat = message.replace('"', '')
        chat = chat.split()
        new_text = []
        stop_words = []
        for token in chat:
            if token not in sw:
                new_text.append(token)
            else:
                stop_words.append(token)
        if (len(new_text) > 0.5 * len(chat) or len(stop_words) > 8) and not len(chat) < 3:
            remove_smalltalk.append(" ".join(chat))

    return {"smalltalkRemoved": remove_smalltalk}


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


@router.post("/spelling_check/",
             summary="Checks for spelling errors in a text.",
             description=
             """
             ## Examples:
             - The sun is shinng brigt today.
             - fr
             - la
             
             ## TODO:
             - See in Latin
             """)
def spelling_check(text: Optional[str] = Form("The sun is shinng brigt today.")):

    words_corpus = words.words()
    brown_corpus = brown.words()
    word_list = set(words_corpus + brown_corpus)
    text_list_lower = text.replace(',', '').replace('.', '').lower().split()
    text_list_original = text.replace(',', '').replace('.', '').split()

    misspelled = []
    for i, _ in enumerate(text_list_lower):
        if text_list_lower[i] not in word_list and text_list_original[i] not in word_list:
            misspelled.append(text_list_original[i])
    if len(misspelled) > 0:
        return {"mistakes": "contains spelling errors"}
    else:
        return {"mistakes": "no spelling errors"}


@router.post("/url_keyword_parser/",
             summary="Checks for spelling errors in a text.",
             description=
             """
             ## Examples:
             - {
    "url": "https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python",
    "includeDomain": True,
    "includeParameter": True,
    "checkValidUrl": True,
    "removeNoneEnglish": False,
    "removeStopwords": True,
    "removeHexLike": True,
    "textSeperator": ", ",
    "splitRegex": "\W",
    "wordWhiteList": None
}
             - fr
             - la
             
             ## TODO:
             - See in Latin
             """)
def url_keyword_parser(url: str = Form("https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python"),
                       includeDomain: bool = Form(True),
                       includeParameter: bool = Form(True),
                       checkValidUrl: bool = Form(True),
                       removeNoneEnglish: bool = Form(False),
                       removeStopwords: bool = Form(True),
                       removeHexLike: bool = Form(True),
                       textSeperator: str = Form(", "),
                       splitRegex: str = Form("\W"),
                       wordWhiteList = Form(None),
                       ):

    """Extract keywords from a url."""
    url = url
    if checkValidUrl and not valid_url(url):
        return ""
    url_obj = urlparse(url)
    keywords = extract_part(url_obj.path,
                            removeNoneEnglish,
                            removeStopwords,
                            removeHexLike,
                            splitRegex,
                            wordWhiteList)
    if includeDomain:
        keywords = (keywords |
                    extract_part(url_obj.netloc,
                                 removeNoneEnglish,
                                 removeStopwords,
                                 removeHexLike,
                                 splitRegex,
                                 wordWhiteList))
    if includeParameter:
        keywords = (keywords |
                    extract_part(url_obj.params,
                                 removeNoneEnglish,
                                 removeStopwords,
                                 removeHexLike,
                                 splitRegex,
                                 wordWhiteList) |
                    extract_part(url_obj.query,
                                 removeNoneEnglish,
                                 removeStopwords,
                                 removeHexLike,
                                 splitRegex,
                                 wordWhiteList) |
                    extract_part(url_obj.fragment,
                                 removeNoneEnglish,
                                 removeStopwords,
                                 removeHexLike,
                                 splitRegex,
                                 wordWhiteList))
    if not textSeperator:
        return " ".join(keywords)
    return textSeperator.join(keywords)

def extract_part(part,
                 removeNoneEnglish: bool = Form(False),
                 removeStopwords: bool = Form(True),
                 removeHexLike: bool = Form(True),
                 splitRegex: str = "\W",
                 wordWhiteList: [] = Form(None),
                 ):
    english_stopwords = set(stopwords.words("english"))
    white_list = set(wordWhiteList)
    split_regex = re.compile(splitRegex)
    english_words = set(words.words())

    if not part:
        return set()

    remaining = set([w.lower() for w in re.split(split_regex, part) if len(w) > 0])
    must_keep = remaining & white_list
    if removeStopwords:
        remaining = remaining - english_stopwords
    if removeNoneEnglish:
        remaining = remaining & english_words
    if removeHexLike:
        remaining = {w for w in remaining if not is_hex(w)}

    return remaining | must_keep

def valid_url(url):
    url_regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    if not url:
        return False
    return re.match(url_regex, url) is not None

def is_hex(part):
    try:
        int(part, 16)
        return True
    except ValueError:
        return False