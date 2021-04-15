"""Just a few utils functions
"""

from typing import Callable
from functools import partial
import spacy
from itertools import chain

def extract_entity(x: str, nlp: Callable) -> list:
    """Generic function to extract entities from an utterance using spacy labelling
    """
    doc = nlp(x)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Monadic function. It's useful to future composing functions (I pretty sure Guido would hate it)
extract_entity_es = partial(extract_entity, nlp=spacy.load("es_core_news_md"))

# It would be more compact using walrus from python 3.8
def decorador(func):
    def wrapper(*args):
        _ = func(*args)
        if _:
          return 1, list(chain(*_))
        else:
          return 0, _
    return wrapper

# Two output function to avoid unnecessary evaluation
decorated_extract_entity_es = decorador(extract_entity_es)
