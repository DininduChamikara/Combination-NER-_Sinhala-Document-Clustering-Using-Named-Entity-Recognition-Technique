import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from fuzzysearch import find_near_matches

nlp = spacy.load("xx_ent_wiki_sm")
fuzzy_threshold = 75
min_token_length = 1


class FuzzyMatcher:
    @staticmethod
    def word_tokenize(text):
        doc = nlp(text)
        tokens = []

        # for white space word tokenize
        for token in doc:
            if len(token.text) > min_token_length:
                tokens.append(token.text)

        return tokens

    @staticmethod
    def tag_entities_from_fuzzy_matcher(text, patterns):
        fuzzy_entities = []
        choices = [el["psttern"] for el in patterns]

        tokens = FuzzyMatcher.word_tokenize(text)

        for token in tokens:
            match = process.extractOne(token, choices, scorer=fuzz.token_sort_ratio)
            if match[1] > fuzzy_threshold:
                match_object = FuzzyMatcher.get_fuzzy_matched_object(
                    patterns, match, token, text
                )
                if match_object:
                    fuzzy_entities.append(match_object)

        return fuzzy_entities

    @staticmethod
    def get_fuzzy_matched_object(patterns, match, token, full_text):
        match_obj = None

        for item in patterns:
            if item["pattern"] == match[0]:
                start_and_end = FuzzyMatcher.get_start_and_end_char(
                    full_text, item["pattern"]
                )
                if start_and_end:
                    match_obj = {
                        "entity": item["label"],
                        "value": item["id"],
                        "text": token,
                        "confidence": match[1],
                        "startChar": start_and_end[0].start,
                        "endChar": start_and_end[0].end,
                        "extractedBy": "fuzzy_matcher",
                    }
        
        return match_obj
    

    @staticmethod
    def get_start_and_end_char(full_text, matched_text):
        near_match = find_near_matches(matched_text, full_text, max_l_dist=1)
        return near_match
