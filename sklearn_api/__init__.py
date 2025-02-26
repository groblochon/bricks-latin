import math
from typing import Optional

import numpy as np
from fastapi import APIRouter
from fastapi import Form
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from Levenshtein import distance
from scipy.spatial.distance import hamming

router = APIRouter()

@router.post("/cosine_similarity/",
             summary="Calculates the cosine similarity between two sentences.",
             description=
             """
             ## Examples:
             - Ten amazing facts about planet Mars., Ten amazing facts about the sun
             - fr
             - la
             """)
def cosine_similarity(text: Optional[str] = Form('Ten amazing facts about planet Mars.'),
                      text2: Optional[str] = Form('Ten amazing facts about the sun')):

    # Transform sentences to a vector
    tfidf = TfidfVectorizer()
    vects = tfidf.fit_transform([text.lower(), text2.lower()])
    vects = vects.todense()
    vect_one, vect_two = np.squeeze(np.asarray(vects[0])), np.squeeze(np.asarray(vects[1]))

    # Calculate the cosine similarity between the two vectors
    try:
        cos_sim = dot(vect_one, vect_two)/(norm(vect_one)*norm(vect_two))
        if cos_sim <= 0.5:
            return {"cosineSimilarity": "Not similar"}
        elif 0.5 < cos_sim < 0.75:
            return {"cosineSimilarity": "Somewhat similar"}
        elif math.isnan(cos_sim):
            return {"cosineSimilarity": "Cannot calculate cosine similarity for empty sentences."}
        else:
            return {"cosineSimilarity": "Very similar"}
    except ValueError:
        return "Cannot determine similarity."


@router.post("/manhattan_distance/",
             summary="Calculates the Manhattan distance between two strings. Uses TF-IDF to vectorize the strings.",
             description=
             """
             ## Examples:
             - The quick brown fox jumps over the lazy dog., The quick yellow cat jumps over the lazy dog.
             - fr
             - la
             """)
def manhattan_distance(text: Optional[str] = Form('The quick brown fox jumps over the lazy dog.'),
                       text2: Optional[str] = Form('The quick yellow cat jumps over the lazy dog.')):

    # Transform sentences to a vector
    tfidf = TfidfVectorizer()
    vects = tfidf.fit_transform([text.lower(), text2.lower()])
    vects = vects.todense()
    vect_one, vect_two = np.squeeze(np.asarray(vects[0])), np.squeeze(np.asarray(vects[1]))

    return {"manhattanDistance": sum(abs(val1-val2) for val1, val2 in zip(vect_one, vect_two))}


@router.post("/levenshtein_distance/",
             summary="Calculates the Manhattan distance between two strings. Uses TF-IDF to vectorize the strings.",
             description=
             """
             ## Examples:
             - John Doe, Jon Doe
             - fr
             - la
             """)
def levenshtein_distance(text: Optional[str] = Form("John Doe"),
                         text2: Optional[str] = Form('Jon Doe'),
                         insertion: Optional[int] = Form(1),
                         deletion: Optional[int] = Form(1),
                         substitution: Optional[int] = Form(1)):

    if insertion is not None:
        weights_tuple = (
            insertion,
            deletion,
            substitution,
        )
        ls_distance = distance(text, text2, weights=weights_tuple)
    else:
        ls_distance = distance(text, text2)
    return {"levenshteinDistance": ls_distance}


@router.post("/hamming_distance/",
             summary="Calculates the Hamming distance between two embeddings to find similar sentences.",
             description=
             """
             ## Examples:
             - Grandpa is eating!, Let's eat, Grandpa!
             - fr
             - la
             """)
def hamming_distance(text: str = Form("Grandpa is eating!"),
                     text2: str = Form("Let's eat, Grandpa!")):

    tfidf = TfidfVectorizer().fit_transform([text, text2])

    dense = tfidf.toarray()
    vect_one, vect_two = np.squeeze(dense[0]), np.squeeze(dense[1])

    if vect_one.shape == () or vect_two.shape == ():
        print("The input vectors are null. Make sure that at least one input is longer than a single word.")

    else:
        hamming_distance = hamming(vect_one, vect_two)
        return {"Hamming distance": hamming_distance}


@router.post("/euclidean_distance/",
             summary="Calculates the euclidean distance between two text embedding vectors.",
             description=
             """
             ## Examples:
             - Grandpa is eating!, Let's eat, Grandpa!
             - fr
             - la
             """)
def euclidean_distance(text: str = Form("Grandpa is eating!"),
                       text2: str = Form("Let's eat, Grandpa!")):

    # Transform sentences to a vector
    tfidf = TfidfVectorizer()
    vects = tfidf.fit_transform([text.lower(), text2.lower()])
    vects = vects.todense()
    vect_one, vect_two = np.squeeze(np.asarray(vects[0])), np.squeeze(np.asarray(vects[1]))

    # Calculate the euclidean distance
    euc_distance = np.linalg.norm(vect_one - vect_two)

    return {"euclidean_distance": euc_distance}