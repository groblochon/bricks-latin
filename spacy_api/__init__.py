import json
import re
from collections import Counter
from heapq import nlargest
from string import punctuation
from typing import Optional, List

import phonenumbers
from fastapi import APIRouter
from fastapi import Form
from spacy.lang.en import STOP_WORDS

from util.utils import LangEnum, SpacySingleton

router = APIRouter()

@router.post("/address_extraction/",
             summary="Extract address using regex",
             description=
"""
## Examples:
- I live at 35 Wood Lane, Pilsbury ME19 7DY, United Kingdom. But I have also lived at 221BE Baker-callum Street, London VIC 3SX, United Kingdom.
- fr
- la

## TODO:
- need to translate regex for all languages
""")
def address_extraction(text: Optional[str] = Form("I live at 35 Wood Lane, Pilsbury ME19 7DY, United Kingdom. But I have also lived at 221BE Baker-callum Street, London VIC 3SX, United Kingdom."),
                       # url: Optional[str] = Form(None),
                       lang: Optional[LangEnum] = Form(LangEnum.EN),
                       # files: Optional[List[UploadFile]] = File(None)
                       ):

    addresses = []
    if text:
        nlp = SpacySingleton.get_nlp(lang)

        doc = nlp(text)

        regex_1 = re.compile(
            r"(?:\d{1,5}(?:[A-Z ]+[ ]?)+(?:[A-Za-z-]+[ ]?)+(?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr(?:\.)?|Rd(?:\.)?|Blvd(?:\.)?|Ln(?:\.)?|St(?:\.)?|Strasse|Hill|Alley|Alle|City)[,](?:[ A-Za-z0-9,]+[ ]?)?)"
        )
        regex_2 = re.compile(
            r"(?:(?:[A-Za-z-]?)+[ ](?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr(?:\.)?|Rd(?:\.)?|Blvd(?:\.)?|Ln(?:\.)?|St(?:\.)?|Strasse|Str(?:\.)?|Hill|Alley|Alle|City)[ ]+\d{1,5},(?:[ A-Za-z0-9,]+[ ]?)?)"
        )

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


@router.post("/color_code_extraction/",
             summary="Extracts CSS colors from a text.",
             description=
             """
             ## Examples:
             - There are more than 42 #colors you could use in CSS, e.g. #ff00ff, hsl(0, 0%, 0%), or rgba(255, 0, 0, 0.3) if you want to use alpha values.
             - fr
             - la
             """)
def color_code_extraction(text: Optional[str] = Form("There are more than 42 #colors you could use in CSS, e.g. #ff00ff, hsl(0, 0%, 0%), or rgba(255, 0, 0, 0.3) if you want to use alpha values."),
                          lang: Optional[LangEnum] = Form(LangEnum.EN)):


    color_codes = []
    if text:
        nlp = SpacySingleton.get_nlp(lang)
        doc = nlp(text)

        # https://developer.mozilla.org/en-US/docs/Web/CSS/color_value
        hexcolor_regex = re.compile(r"#([0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})(?![0-9a-fA-F])")
        rgb_regex = re.compile(r"(rgba|rgb)\([^\)]*\)")
        hsl_regex = re.compile(r"(hsla|hsl)\([^\)]*\)")
        hwb_regex = re.compile(r"hwb\([^\)]*\)")

        for regex in [hexcolor_regex, rgb_regex, hsl_regex, hwb_regex]:
            for match in regex.finditer(text):
                start, end = match.span()
                span = doc.char_span(start, end, alignment_mode="expand")
                color_codes.append(["color", span.start, span.end])

    return {"extractedColorCodes": color_codes}


@router.post("/date_extraction/",
             summary="Detects dates in a text and returns them in a list.",
             description=
             """
             ## Examples:
             - Today is 04.11.2022. Yesterday was 03/11/2022. Tomorrow is 05-11-2022. Day after tomorrow is 6 Nov 2022.
             - fr
             - la
             
             ## TODO:
             - translate regex
             """)
def date_extraction(text: Optional[str] = Form("Today is 04.11.2022. Yesterday was 03/11/2022. Tomorrow is 05-11-2022. Day after tomorrow is 6 Nov 2022."),
                    lang: Optional[LangEnum] = Form(LangEnum.EN)):

    dates = []
    if text:
        nlp = SpacySingleton.get_nlp(lang)
        doc = nlp(text)
        regex = re.compile(
            r"(?:[0-9]{1,2}|[0-9]{4}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\/\. -]{1}(?:[0-9]{1,2}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\/\. -]{1}(?:[0-9]{2,4})"
        )

        for match in regex.finditer(text):
            start, end = match.span()
            span = doc.char_span(start, end, alignment_mode="expand")
            dates.append(["date", span.start, span.end])

    return {"dates": dates}


@router.post("/time_extraction/",
             summary="Extracts times from a given text.",
             description=
             """
             ## Examples:
             - Right now it is 14:40:37. Three hours ago it was 11:40 am. Two hours and twenty mins from now it will be 5PM.
             - fr
             - la
             """)
def time_extraction(text: Optional[str] = Form("Right now it is 14:40:37. Three hours ago it was 11:40 am. Two hours and twenty mins from now it will be 5PM."),
                    lang: Optional[LangEnum] = Form(LangEnum.EN)):

    times = []
    if text:
        nlp = SpacySingleton.get_nlp(lang)
        doc = nlp(text)
        regex = re.compile(
            r"\b(1[0-2]|[1-9])\s*[apAP][. ]*[mM]\.?|(?:(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?(?:(?:\s?[ap](?:\.m\.)?)|(?:\s?[AP](?:\.M\.)?)))|(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?"
        )

        for match in regex.finditer(text):
            start, end = match.span()
            span = doc.char_span(start, end, alignment_mode="expand")
            times.append(["time", span.start, span.end])


@router.post("/gazetteer_extraction/",
             summary="Detects full entities in a text based on some hints.",
             description=
             """
             ## Examples:
             - Max Mustermann decided to join Kern AI, where he wants to build great software.
             - fr
             - la
             """)
def gazetteer_extraction(text: Optional[str] = Form("Max Mustermann decided to join Kern AI, where he wants to build great software."),
                         lang: Optional[LangEnum] = Form(LangEnum.EN),
                         lookup_values: List[str] = Form(["Max", "Leon", "Kai", "Aaron"]),
                         your_label: str = Form("person")):

    """Detects full entities in a text based on some hints."""
    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    matches = []

    for chunk in doc.noun_chunks:
        if any(
                [chunk.text in trie or trie in chunk.text for trie in lookup_values]
        ):
            yield your_label, chunk.start, chunk.end
    return {f"{your_label}s": matches}


@router.post("/regex_extraction/",
             summary="Detects regex matches in a given text.",
             description=
             """
             ## Examples:
             - Check out https://kern.ai!
             - fr
             - la
             """)
def regex_extraction(text: Optional[str] = Form("Check out https://kern.ai!"),
                         lang: Optional[LangEnum] = Form(LangEnum.EN),
                         regex: str = Form("https:\/\/[a-zA-Z0-9.\/]+"),
                         your_label: str = Form("url")):

    """Detects regex matches in a given text."""
    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    matches = []

    def regex_search(pattern, string):
        """
        some helper function to easily iterate over regex matches
        """
        prev_end = 0
        while True:
            match = re.search(pattern, string)
            if not match:
                break

            start, end = match.span()
            yield start + prev_end, end + prev_end

            prev_end += end
            string = string[end:]

    for start, end in regex_search(regex, text):
        span = doc.char_span(start, end, alignment_mode="expand")
        matches.append([your_label, span.start, span.end])

    return {f"{your_label}s": matches}


@router.post("/window_search_extraction/",
             summary="Searches for a given list of words in a given text and returns the surrounding noun chunks.",
             description=
             """
             ## Examples:
             - Max Mustermann decided to join Kern AI, where he wants to build great software.
             - fr
             - la
             """)
def window_search_extraction(text: Optional[str] = Form("Max Mustermann decided to join Kern AI, where he wants to build great software."),
                             lang: Optional[LangEnum] = Form(LangEnum.EN),
                             lookup_values: List[str] = Form(["join", "works at", "is employed by"]),
                             window_size: str = Form(6),
                             your_label: str = Form("person")):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    matches = []
    for chunk in doc.noun_chunks:
        left_bound = max(chunk.sent.start, chunk.start - (window_size // 2) + 1)
        right_bound = min(chunk.sent.end, chunk.end + (window_size // 2) + 1)
        window_doc = doc[left_bound:right_bound]
        if any([term in window_doc.text for term in lookup_values]):
            matches.append([your_label, chunk.start, chunk.end])

    return {f"{your_label}s": matches}


@router.post("/work_of_art_extraction/",
             summary="Extracts the name of the book from a text.",
             description=
             """
             ## Examples:
             - The bestseller of last month is "Mystery of the Floridian Porter" by John Doe.
             - fr
             - la
             """)
def work_of_art_extraction(text: Optional[str] = Form('The bestseller of last month is "Mystery of the Floridian Porter" by John Doe.'),
                           lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    found = []

    for entity in doc.ents:
        if entity.label_ == "WORK_OF_ART":
            found.append(["work of art", entity.start, entity.end])

    return {"works of art": found}


@router.post("/bic_extraction/",
             summary="Extracts BIC from text.",
             description=
             """
             ## Examples:
             - My BIC number is COBADEBBXXX
             - fr
             - la
             """)
def bic_extraction(text: Optional[str] = Form('My BIC number is COBADEBBXXX'),
                   lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    regex = re.compile(r'\b[A-Z0-9]{4,4}[A-Z]{2,2}[A-Z2-9][A-NP-Z0-9]([X]{3,3}|[A-WY-Z0-9]{1,1}[A-Z0-9]{2,2}|\s|\W|$)')
    bic = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        bic.append(["BIC", span.start, span.end])
    return {"bic": bic}


@router.post("/credit_card_extraction/",
             summary="Extracts the credit/debit card number from a text.",
             description=
             """
             ## Examples:
             - This is my card details please use it carefully 4569-4039-6101-4710.
             - fr
             - la
             """)
def bic_extraction(text: Optional[str] = Form('This is my card details please use it carefully 4569-4039-6101-4710.'),
                   lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    regex = re.compile(r"(\d{4}[-\s]?){3}\d{3,4}")

    credit = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        credit.append([span.start, span.end, span.text])

    return {"creditCard": credit}


@router.post("/digit_extraction/",
             summary="Extracts digits of variable length.",
             description=
             """
             ## Examples:
             - My PIN is 1337.
             - fr
             - la
             """)
def digit_extraction(text: Optional[str] = Form('My PIN is 1337.'),
                     lang: Optional[LangEnum] = Form(LangEnum.EN),
                     digit_length: int = Form(4)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    num_string = "{"+f"{digit_length}"+"}"
    regex = re.compile(rf"(?<![0-9])[0-9]{num_string}(?![0-9])")

    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        return {"Number": [span.start, span.end]}


@router.post("/iban_extraction/",
             summary="Extracts IBAN from text",
             description=
             """
             ## Examples:
             - DE89370400440532013000.
             - fr
             - la
             """)
def iban_extraction(text: Optional[str] = Form('DE89370400440532013000'),
                     lang: Optional[LangEnum] = Form(LangEnum.EN)):
    """Extracts IBAN from text"""

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    regex = re.compile(r"[A-Z]{2}\d{2} ?\d{4} ?\d{4} ?\d{4} ?\d{4} ?[\d]{0,2}")

    isbn = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        isbn.append(["IBAN", span.start, span.end])
    return {"iban": isbn}


@router.post("/ip_extraction/",
             summary="Extracts IP addresses from text",
             description=
             """
             ## Examples:
             - The IP addressing range is from 0.0.0.0 to 255.255.255.255.
             - fr
             - la
             """)
def iban_extraction(text: Optional[str] = Form('The IP addressing range is from 0.0.0.0 to 255.255.255.255'),
                    lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    regex = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

    ip_addresses = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        ip_addresses.append(["ip_address", span.start, span.end])

    return {"ip_addresses": ip_addresses}


@router.post("/percentage_extraction/",
             summary="Extracts the Percentages from a text",
             description=
             """
             ## Examples:
             - percentages 110% are found -.5% at 42,13% positions 1, 5 and 8.
             - fr
             - la
             """)
def percentage_extraction(text: Optional[str] = Form('percentages 110% are found -.5% at 42,13% positions 1, 5 and 8'),
                          lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    regex = re.compile(r"(-?\d+(?:[.,]\d*)?|-?[.,]\d+)\s*%")
    percentages = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        percentages.append(["percentage", span.start, span.end])
    return {"percentages": percentages}


@router.post("/phone_number_extraction/",
             summary="Extracts the Percentages from a text",
             description=
             """
             ## Examples:
             - So here's my number +442083661177. Call me maybe!
             - fr
             - la
             """)
def phone_number_extraction(text: Optional[str] = Form('So heres my number +442083661177. Call me maybe!'),
                            lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    regex = re.compile(r"[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}")
    regex.findall(text)

    valid_numbers = []
    for match in regex.finditer(text):
        try:
            parsed_num = phonenumbers.parse(match.group(0), None)
            if phonenumbers.is_valid_number(parsed_num):
                start, end = match.span()
                span = doc.char_span(start, end)
                valid_numbers.append(["phoneNumber", span.start, span.end])
        except phonenumbers.phonenumberutil.NumberParseException:
            pass

    return {"phoneNumbers": valid_numbers}


@router.post("/price_extraction/",
             summary="Extracts prices from a given text.",
             description=
             """
             ## Examples:
             - A desktop with i7 processor costs 950 dollars in the US.
             - fr
             - la
             """)
def price_extraction(text: Optional[str] = Form('A desktop with i7 processor costs 950 dollars in the US.'),
                     lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    prices = []
    for entity in doc.ents:
        if entity.label_ == "MONEY":
            prices.append(["price", entity.start, entity.end])

    return {"prices": prices}


@router.post("/filepath_extraction/",
             summary="Extracts a path from a string.",
             description=
             """
             ## Examples:
             - My favourite file is stored in: /usr/bin/myfavfiles/cats.png
             - fr
             - la
             """)
def filepath_extraction(text: Optional[str] = Form('My favourite file is stored in: /usr/bin/myfavfiles/cats.png'),
                        lang: Optional[LangEnum] = Form(LangEnum.EN),
                        separator: str = Form("/"),
                        your_label: str = Form("path")):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    # Extracts the paths from the texts
    paths = [x for x in text.split() if len(x.split(separator)) > 1]

    # We need to add an \ before separators to use them in regex
    regex_paths = [i.replace(separator, "\\"+separator) for i in paths]
    print(regex_paths)

    matches = []
    for path in regex_paths:
        pattern = rf"({path})"
        match = re.search(pattern, text)

        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")

        matches.append([your_label, span.start, span.end])

    return {f"{your_label}s": matches}


@router.post("/url_extraction/",
             summary="Extracts urls from a given text.",
             description=
             """
             ## Examples:
             - Check out https://kern.ai!
             - fr
             - la
             """)
def url_extraction(text: Optional[str] = Form('Check out https://kern.ai!'),
                   lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    regex_pattern = re.compile(r"(?:(?:(?:https?|ftp):\/\/){1})?[\w\-\/?=%.]{3,}\.[\/\w\-&?=%.]{2,}")
    regex_pattern.findall(text)

    urls = []
    for match in regex_pattern.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        urls.append(["url", span.start, span.end])

    return {"urls": urls}


@router.post("/email_extraction/",
             summary="Extracts urls from a given text.",
             description=
             """
             ## Examples:
             - If you have any questions, please contact johannes.hoetter@kern.ai.
             - fr
             - la
             """)
def email_extraction(text: Optional[str] = Form('If you have any questions, please contact johannes.hoetter@kern.ai.'),
                     lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    regex = re.compile(r"([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)")

    emails = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        emails.append(["email", span.start, span.end])

    return {"emails": emails}


@router.post("/location_extraction/",
             summary="Uses SpaCy to extract locations from a text.",
             description=
             """
             ## Examples:
             - Tokyo is a beautiful city, which is not located in Kansas, USA.
             - fr
             - la
             """)
def location_extraction(text: Optional[str] = Form('Tokyo is a beautiful city, which is not located in Kansas, USA.'),
                        lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    names = []
    for ent in doc.ents:
        if ent.label_ == "GPE" or ent.label_ == "LOC":
            names.append(["location", ent.start, ent.end])
    return {"locations": names}


@router.post("/person_extraction/",
             summary="Returns the occurrences of names of people in a dictionary.",
             description=
             """
             ## Examples:
             - John Doe worked with Jane Doe and now they are together.
             - fr
             - la
             """)
def person_extraction(text: Optional[str] = Form('John Doe worked with Jane Doe and now they are together.'),
                      lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    names = []

    for entity in doc.ents:
        if entity.label_ == "PERSON":
            names.append(["person", entity.start, entity.end])
    # "name" will contain all the occurrences of a particular name.
    # This is because spacy treats each word in a text as a unique vector.
    # So, two occurrences of "Div" does not mean "Div" == "Div"!
    names = {"names": names}

    return names

# Load JSON file with all zip code regex patterns
with open('spacy_api/zip_codes.json') as f:
    zip_codes_json = json.load(f)

@router.post("/zipcode_extraction/",
             summary="Extracts a zipcode from a string using regex.",
             description=
             """
             ## Examples:
             - 10 Downing Street London SW1A 2AA
             - fr
             - la
             """)
def zipcode_extraction(text: Optional[str] = Form('10 Downing Street London SW1A 2AA'),
                       lang: Optional[LangEnum] = Form(LangEnum.EN),
                       country_id: str = Form("GB")):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    match = re.search(zip_codes_json[country_id], text)

    try:
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
    except AttributeError:
        return "No zipcodes found"

    return {country_id: ["zipcode", span.start, span.end]}


@router.post("/hashtag_extraction/",
             summary="Detects hashtags in a text and returns them in a list.",
             description=
             """
             ## Examples:
             - In tech industry, #devrel is a very hot topic.
             - fr
             - la
             """)
def hashtag_extraction(text: Optional[str] = Form('In tech industry, #devrel is a very hot topic.'),
                       lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    regex = re.compile(r"#(\w*)")
    regex.findall(text)

    hashtags = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        hashtags.append(["hashtag", span.start, span.end])

    return {"hashtags": hashtags}


@router.post("/noun_match_extraction/",
             summary="Extracts all similar noun chunks from a text",
             description=
             """
             ## Examples:
             - Leo likes tasty pizza. Mary loves delicious cake. And Moritz loves tasty bread.
             - fr
             - la
             """)
def noun_match_extraction(text: Optional[str] = Form('Leo likes tasty pizza. Mary loves delicious cake. And Moritz loves tasty bread.'),
                          lang: Optional[LangEnum] = Form(LangEnum.EN)):

    """Extracts all similar noun chunks from a text"""
    # instantiate empty lists to store already encountered words and for found matches
    word_repo = []
    matches = []

    # get noun chunks from spacy
    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text.lower())
    nc = [i.text.lower() for i in doc.noun_chunks]

    # loop through all noun chunks
    for noun_chunk in nc:
        print(noun_chunk)
        # if noun chunk has more than one word, take first word as a target word
        if len(noun_chunk.split()) >= 2:
            target_word = noun_chunk.split()[0]

            # if target word has been used before, stop process
            if target_word in word_repo:
                pass
            else:
                # pass word to repository to avoid duplicate use
                word_repo.append(target_word)

                # create regex_pattern with target word
                pattern = rf"\W*({target_word})\W*([^\s]+)"

                # extract the spans of all found matches
                for item in re.finditer(pattern, text):
                    start, end = item.span()
                    span = doc.char_span(start, end, alignment_mode="expand")
                    matches.append(["match", span.start, span.end])
        else:
            pass

    return {"quote": matches}


@router.post("/org_extraction/",
             summary="Detects organizations in a given text.",
             description=
             """
             ## Examples:
             - We are developers from Kern.ai.
             - fr
             - la
             """)
def org_extraction(text: Optional[str] = Form('We are developers from Kern.ai.'),
                   lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    organisations = []

    for entity in doc.ents:
        if entity.label_ == "ORG":
            organisations.append(["org", entity.start, entity.end])

    return {"organisations": organisations}


@router.post("/part_of_speech_extraction/",
             summary="Yields POS tags using spaCy.",
             description=
             """
             ## Examples:
             - My favourite british tea is Yorkshire tea.
             - fr
             - la
             """)
def part_of_speech_extraction(text: Optional[str] = Form('My favourite british tea is Yorkshire tea.'),
                              lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    pos_tags = []
    for token in doc:
        pos = token.pos_

        start, end = token.i, token.i +1
        span = doc.char_span(start, end, alignment_mode="expand")

        pos_tags.append([pos, span.start, span.end])

    return {"POS tags": pos_tags}



@router.post("/quote_extraction/",
             summary="Extracts all the quotes from a text.",
             description=
             """
             ## Examples:
             - "Hello, Nick," said Harry.
               "Hello, hello," said Nearly Headless Nick, starting and looking round. He wore a dashing, plumed hat on his long curly hair, and a tunic with a ruff, which concealed the fact that his neck was almost completely severed. He was pale as smoke, and Harry could see right through him to the dark sky and torrential rain outside.
               "You look troubled, young Potter," said Nick, folding a transparent letter as he spoke and tucking it inside his doublet.
               "So do you," said Harry.
             - fr
             - la
             """)
def quote_extraction(text: Optional[str] = Form('''"Hello, Nick," said Harry.
                                                "Hello, hello," said Nearly Headless Nick, starting and looking round. He wore a dashing, plumed hat on his long curly hair, and a tunic with a ruff, which concealed the fact that his neck was almost completely severed. He was pale as smoke, and Harry could see right through him to the dark sky and torrential rain outside.
                                                "You look troubled, young Potter," said Nick, folding a transparent letter as he spoke and tucking it inside his doublet.
                                                "So do you," said Harry.'''),
                   lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    regex = re.compile(r'\"(.+?)"|\'(.*?)\'')

    quotes = []
    for match in regex.finditer(text):
        start, end = match.span()
        span = doc.char_span(start, end, alignment_mode="expand")
        quotes.append(["quote", span.start, span.end])

    return {"quote": quotes}


@router.post("/substring_extraction/",
             summary="Extracts a common substring between two strings.",
             description=
             """
             ## Examples:
             - Hello this is my flat. This is a duplicate.
             - fr
             - la
             """)
def substring_extraction(text: Optional[str] = Form('Hello this is my flat. This is a duplicate.'),
                         lang: Optional[LangEnum] = Form(LangEnum.EN),
                         substring: str = Form("This is a duplicate.")):

    start_index = text.find(substring)
    end_index = start_index + len(substring)

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    if start_index != -1:
        span = doc.char_span(start_index, end_index, alignment_mode="expand")
        return {"Substring": [span.start, span.end]}
    else:
        return "No substring found!"


@router.post("/spacy_lemmatizer/",
             summary="Converts words in a sentence to there base form.",
             description=
             """
             ## Examples:
             - Hello, I am talking about coding at Kern AI!
             - fr
             - la
             """)
def spacy_lemmatizer(text: Optional[str] = Form('Hello, I am talking about coding at Kern AI!'),
                     lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    final_text = ""
    for i, token in enumerate(doc):
        if i > 0:
            diff = token.idx - (doc[i - 1].idx + len(doc[i - 1]))
            if diff > 0:
                final_text += " " * diff
        final_text += token.lemma_
    return {"lemmatized_text": final_text}


@router.post("/noun_splitter/",
             summary="Creates embedding chunks based on the nouns in a text",
             description=
             """
             ## Examples:
             - My favorite noun is 'friend'.
             - fr
             - la
             """)
def noun_splitter(text: Optional[str] = Form("My favorite noun is 'friend'."),
                  lang: Optional[LangEnum] = Form(LangEnum.EN)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    nouns_sents = set()
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "NOUN" and len(token.text) > 1:
                nouns_sents.add(token.text)

    return {"nouns": list(nouns_sents)}


@router.post("/text_summarization/",
             summary="Generates the summary of a lengthy text.",
             description=
             """
             ## Examples:
             - There was a time when he would have embraced the change that was coming. In his youth, he sought 
               adventure and the unknown, but that had been years ago. He wished he could go back and learn to find the 
               excitement that came with change but it was useless. That curiosity had long left him to where he had come to 
               loathe anything that put him out of his comfort zone.
             - fr
             - la
             
             ## TODO:
             - add latin stopwords
             """)
def text_summarization(text: Optional[str] = Form("""There was a time when he would have embraced the change that was coming. In his youth, he sought 
                                                    adventure and the unknown, but that had been years ago. He wished he could go back and learn to find the 
                                                    excitement that came with change but it was useless. That curiosity had long left him to where he had come to 
                                                    loathe anything that put him out of his comfort zone."""),
                       lang: Optional[LangEnum] = Form(LangEnum.EN),
                       length: float = Form(0.5)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)
    word_count = {}

    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_count.keys():
                    word_count[word.text] = 1
                else:
                    word_count[word.text] += 1

    maximum_count = max(word_count.values())

    for word in word_count.keys():
        word_count[word] = word_count[word] / maximum_count

    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}

    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_count.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_count[word.text.lower()]
                else:
                    sentence_scores[sent] += word_count[word.text.lower()]

    size = int(len(sentence_tokens) * length)
    extracted_sentences = nlargest(size, sentence_scores, key=sentence_scores.get)
    summarise = [word.text for word in extracted_sentences]
    summary = " ".join(summarise)

    return {"summary": summary}


@router.post("/most_frequent_words/",
             summary="Generates the frequency of the words and shows top n words",
             description=
             """
             ## Examples:
             - APPL went down by 5% in the past two weeks. Shareholders are concerned over the continued recession since APPL and NASDAQ have been hit hard by this recession. Risks pertaining to short-selling are pouring in as APPL continues to depreciate. If the competitors come together and start short-selling, the stock can face calamity.
             - fr
             - la
             """)
def most_frequent_words(text: Optional[str] = Form("APPL went down by 5% in the past two weeks. Shareholders are concerned over the continued recession since APPL and NASDAQ have been hit hard by this recession. Risks pertaining to short-selling are pouring in as APPL continues to depreciate. If the competitors come together and start short-selling, the stock can face calamity."),
                        lang: Optional[LangEnum] = Form(LangEnum.EN),
                        n_words: int = Form(5)):

    nlp = SpacySingleton.get_nlp(lang)
    doc = nlp(text)

    words = [token.text for token in doc if not token.is_stop and not token.is_punct]

    return {"frequentWords": Counter(words).most_common(n_words)}