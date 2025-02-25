import spacy
from enum import Enum

class LangEnum(str, Enum):
    AUTO = 'AUTO'
    EN = 'en_core_web_sm'
    FR = 'fr_core_news_sm'
    LA = 'la_core_web_lg'

class SpacySingleton:
    nlps = { LangEnum.FR : None,
            LangEnum.EN : None,
            LangEnum.LA : None
           }

    @classmethod
    def get_nlp(cls, lang: LangEnum):
        nlp = cls.nlps[lang]
        if nlp is None:
            nlp = spacy.load(lang.value)
        return nlp