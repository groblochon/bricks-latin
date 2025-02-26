from typing import Optional

import tiktoken
from fastapi import APIRouter
from fastapi import Form

router = APIRouter()

@router.post("/tiktoken_length_classifier/",
             summary="Uses the Tiktoken library to count tokens in a string.",
             description=
             """
             ## Examples:
             - The sun is shining bright today.
             - fr
             - la
             """)
def tiktoken_length_classifier(text: Optional[str] = Form('The sun is shining bright today.'),
                               encoding_model: str = Form("cl100k_base")):

    encoding = tiktoken.get_encoding(encoding_model)
    tokens = encoding.encode(text)
    num_tokens = len(tokens)

    if num_tokens < 128:
        return {"token_length": "Short"}
    elif num_tokens < 1024:
        return {"token_length": "Medium"}
    else:
        return{"token_length": "Long"}


@router.post("/tiktoken_token_counter/",
             summary="Uses the Tiktoken library to count tokens in a string.",
             description=
             """
             ## Examples:
             - What a beautiful day to count tokens.
             - fr
             - la
             """)
def tiktoken_token_counter(text: Optional[str] = Form('What a beautiful day to count tokens.'),
                           encoding_model: str = Form("cl100k_base")):

    encoding = tiktoken.get_encoding(encoding_model)
    tokens = encoding.encode(text)
    return {"token_length": len(tokens)}