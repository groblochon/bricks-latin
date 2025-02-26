from typing import Optional
from fastapi import APIRouter
from fastapi import Form
import requests

router = APIRouter()

@router.post("/question_type_classifier/",
             summary="Uses custom E5 model to classify the question type of a text.",
             description=
             """
             ## Examples:
             - Sushi restaurants Barcelona
             - fr
             - la
             """)
def question_type_classifier(text: Optional[str] = Form('Sushi restaurants Barcelona.'),
                             model_name: str = Form('KernAI/multilingual-e5-question-type')):

    payload = {
        "model_name": model_name,
        "text": text
    }
    response = requests.post("https://free.api.kern.ai/inference", json=payload)
    if response.ok:
        return {"question_type": response.json()["label"]}
    return response.raise_for_status()


@router.post("/communication_style_classifier/",
             summary="Uses custom E5 model to classify communication style of a text.",
             description=
             """
             ## Examples:
             - Change the number in row 2 and 3.
             - fr
             - la
             """)
def communication_style_classifier(text: Optional[str] = Form('Change the number in row 2 and 3.'),
                                   model_name: str = Form('KernAI/multilingual-e5-communication-style')):

    payload = {
        "model_name": model_name,
        "text": text
    }
    response = requests.post("https://free.api.kern.ai/inference", json=payload)
    if response.ok:
        return {"communication_style": response.json()["label"]}
    return response.raise_for_status()