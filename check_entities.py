import spacy
from spacy.lang.si import Sinhala


nlp = Sinhala()
ruler = nlp.add_pipe("entity_ruler")

nlp_ner = spacy.load("./SpacyNER/model-best")

class ExtractEntities:

    @staticmethod
    def tag_fuzzy_entities_from_text(text):
        