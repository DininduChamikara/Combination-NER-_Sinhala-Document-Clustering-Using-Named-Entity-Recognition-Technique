from multiprocessing import freeze_support
from simpletransformers.ner import NERModel, NERArgs
import spacy

# for fuzzy
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json

# evaluate model
from nervaluate import Evaluator


with open("./SpacyNER/annotations_v1.0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("./FuzzyMatcher/dict_sinhala.json", "r", encoding="utf-8") as f:
    dictionary = json.load(f)

with open("./FuzzyMatcher/stopwords.json", "r", encoding="utf-8") as s:
    stopwords = json.load(s)

confidenceLevels = pd.DataFrame(
    {
        "wordLength": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "minRatio": [100, 100, 100, 100, 80, 80, 80, 80, 75, 75],
    }
)

testingDataPortionDivider = 5 # 20% => 100/5 => 5

def trueOutputsArrGenerator(annotations):
    trueOutputs = []
    testData = annotations[0: int(len(annotations)/testingDataPortionDivider)]
    for annot in testData:
        trueOutputs.append(trueOutputGenerator(annot))
    return trueOutputs

def predOutputsArrGenerator(annotations, spacyNER, bertNER):
    predOutputs = []
    testData = annotations[0: int(len(annotations)/testingDataPortionDivider)]
    for annot in testData:
        nerResult = namedEntityRecognition(annot[0], spacyNER, bertNER)
        predOutputs.append(predOutputGenerator(annot[0], nerResult))
    return predOutputs

def getFuzzyRatio(token=None, confidence_levels=True, default_level=85):
    # check for the appropriate formats
    assert isinstance(token, str), "Tokens can be str() type only"
    # We check if confidence levels are set
    if confidence_levels:
        for i, row in confidenceLevels.iterrows():
            if len(token) > confidenceLevels["wordLength"].max():
                min_ratio = confidenceLevels["minRatio"].min()
            else:
                min_ratio = confidenceLevels.loc[
                    confidenceLevels["wordLength"] == len(token)
                ]["minRatio"].values[0]

    # Fallback if confidence levels are not set
    else:
        min_ratio = default_level
    return int(min_ratio)


def getFuzzySimilarity(token=None, dictionary=None, min_ratio=None):
    # Check for appropriate formats
    assert isinstance(token, str), "Tokens can be str() type only"
    assert isinstance(
        dictionary, dict
    ), "Dictionary format should be provided in the dictionary parameter."
    assert isinstance(
        min_ratio, int
    ), "Integer format should be provided in the minimum-ratio parameter."

    for key, values in dictionary.items():
        # Using the process option of FuzzyWuzzy, we can search through the entire dictionary for the best match
        match = process.extractOne(token, values, scorer=fuzz.ratio)
        # Match is a tuple with the match value and the similary score.
        if min_ratio <= match[1]:
            return match + (key,)


def get_word_indices(sentence):
    # Split the sentence into words using whitespace and punctuation as delimiters
    words = sentence.split()
    # Initialize a list to store word indices
    word_indices = []
    start = 0
    for word in words:
        end = start + len(word)
        word_indices.append([word, start, end])
        start = end + 1  # Add 1 for the space between words
    return word_indices


def predOutputGenerator(str, modelResults):
    predOut = []
    wordIndices = get_word_indices(str)
    for indice in wordIndices:
        for word in modelResults:
            if word[0] == indice[0]:
                predOutEnt = {"label": word[1], "start": indice[1], "end": indice[2]}
                predOut.append(predOutEnt)
    return predOut

def trueOutputGenerator(annotationObject):
    trueOut = []
    for ent in annotationObject[1]["entities"]:
        trueOutEnt = {"label": ent[2], "start": ent[0], "end": ent[1]}
        trueOut.append(trueOutEnt)
    return trueOut


label = ["LOCATION", "OTHER", "PERSON", "DATE", "ORGANIZATION", "TIME"]

args = NERArgs()
args.num_train_epochs = 10
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

def removeStopWords(sentence):
    result = sentence.split()
    final = []
    for i in result:
        if i not in stopwords:
            final.append(i)
            final.append(" ")
    stop_removed = "".join(final)
    return stop_removed

def namedEntityRecognition (sentence, spacyNER, bertNER):
    stopRemovedSentence = removeStopWords(sentence)
    doc_spacy_ner = spacyNER(stopRemovedSentence)
    prediction, model_output = bertNER.predict([stopRemovedSentence]) 

    tokens = stopRemovedSentence.split()

    fuzzyResults = fuzzyResultGenerator(tokens)

    final_entities = set()

    # unioin start #
    # BERT
    for entity in prediction[0]:
        for key, value in entity.items():
            if value not in final_entities:
                if value != "OTHER":
                    unique_entity_BERT = (key, value)
                    if unique_entity_BERT not in final_entities:
                        final_entities.add(unique_entity_BERT)
    # SPACY
    for spacyEnt in doc_spacy_ner.ents:
        if spacyEnt.text != "":
            unique_entity_SPACY = (
                spacyEnt.text,
                spacyEnt.label_
            )
            if unique_entity_SPACY not in final_entities:
                final_entities.add(unique_entity_SPACY)
    # FUZZY
    for fuzzEnt in fuzzyResults:
        unique_entity_FUZZY = (
            fuzzEnt[0],
            fuzzEnt[1],
        )
        if unique_entity_FUZZY not in final_entities:
            final_entities.add(unique_entity_FUZZY)
    # unioin end #

    return final_entities



def fuzzyResultGenerator(tokens):
    fuzzyResults = set()
    for token in tokens:
        fuzzy_ratio = getFuzzyRatio(token=token, confidence_levels=True)
        similarity_score = getFuzzySimilarity(
            token=token, dictionary=dictionary, min_ratio=fuzzy_ratio
        )
        if not similarity_score == None:
            fuzzyEntity = (token, similarity_score[2])
            fuzzyResults.add(fuzzyEntity)
    return fuzzyResults


if __name__ == "__main__":
    freeze_support()

    nlp_ner = spacy.load("./SpacyNER/model-best")

    model = NERModel(
        "bert", "./BERT_NER/trained_model", labels=label, args=args, use_cuda=False
    )

    trueOutputsArr = trueOutputsArrGenerator(data['annotations'])
    predOutputsArr = predOutputsArrGenerator(data['annotations'], nlp_ner, model)

    evaluator = Evaluator(trueOutputsArr, predOutputsArr, tags=['LOCATION', 'PERSON', 'ORGANIZATION', 'DATE', 'TIME'])

    # Returns overall metrics and metrics for each tag
    results, results_per_tag = evaluator.evaluate()

    print(results)
    print(results_per_tag)
