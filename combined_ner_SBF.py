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


testingDataPortionDivider = 100 # 20% => 100/5 => 5


def trueOutputsArrGenerator(annotations):
    trueOutputs = []
    testData = annotations[0: int(len(annotations)/testingDataPortionDivider)]
    
    for annot in testData:
        trueOutputs.append(trueOutputGenerator(annot))

    return trueOutputs

# def predOutputsArrGenerator(annotations, modelResults):

#     predOutputs = []
#     testData = annotations[0: int(len(annotations)/testingDataPortionDivider)]

#     for annot in testData:
#         predOutputs.append(predOutputGenerator(annot[0], modelResults))

#     return predOutputs

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

testString = "2023-2025 ලෝක ටෙස්ට් ශූරතා තරගාවලිය එළඹෙන ජූනි මාසයේ දී ආරම්භ වීමට නියමිත යි. ඒ සඳහා සහභාගි වන එක් එක් කණ්ඩායමට හිමි වන ටෙස්ට් තරග ප්‍රමාණය මේ වන විට ප්‍රකාශයට පත් කර අවසන්. ඒ අනුව මෙවර තරගාවලිය යටතේ ද ශ්‍රී ලංකාවට ටෙස්ට් තරග 12කට ක්‍රීඩා කිරීමේ අවස්ථාව ලැබෙනවා. එයින් ටෙස්ට් තරග 6ක් ශ්‍රී ලංකාවේ ද ඉතිරි ටෙස්ට් තරග 6 විදෙස් රටවල්වල දී ද පැවැත්වීමට නියමිත යි.අවසන් වරට පැවති 2021-2023 ලෝක ටෙස්ට් ශූරතා තරගාවලියේ දී ශ්‍රී ලංකාව සැලකිය යුතු තරම් දස්කම් පෙන්වීමට සමත් වුණා. එනිසා මෙවර ලෝක ටෙස්ට් ශූරතාවේ අවසන් තරගය සඳහා පිවිසීමට ශ්‍රී ලංකාවට තිබෙන අවස්ථාව පිළිබඳව යම් විශ්ලේෂණයක් මෙලෙසින් ඔබ වෙත ගෙන එනවා."

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


    # print("PREDICTIONS")
    # print(prediction)

    # print("DOC SPACY NER ENTS")
    # print(doc_spacy_ner.ents)

    # print("FUZZY RESULTS")
    # print(fuzzyResultGenerator(tokens))

    # print(final_entities)

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

    doc_spacy = nlp_ner(testString)

    model = NERModel(
        "bert", "./BERT_NER/trained_model", labels=label, args=args, use_cuda=False
    )

    prediction, model_output = model.predict([testString])

    # print("PREDICTIONS")
    # print(prediction)

    # namedEntityRecognition(testString, nlp_ner, model)

    # stop words remove
    result = testString.split()
    final = []

    for i in result:
        if i not in stopwords:
            final.append(i)
            final.append(" ")
    stop_removed = "".join(final)

    tokens = stop_removed.split()

    dictionary = dictionary

    fuzzyResults = set()

    for token in tokens:
        fuzzy_ratio = getFuzzyRatio(token=token, confidence_levels=True)
        similarity_score = getFuzzySimilarity(
            token=token, dictionary=dictionary, min_ratio=fuzzy_ratio
        )
        if not similarity_score == None:
            fuzzyEntity = (token, similarity_score[2])
            fuzzyResults.add(fuzzyEntity)

    final_entities = set()
    intersecResults = set()

    for entity in prediction[0]:
        for key, value in entity.items():
            for ent in doc_spacy.ents:
                for fuzzEnt in fuzzyResults:
                    if (
                        key == ent.text
                        and value == ent.label_
                        and key == fuzzEnt[0]
                        and value == fuzzEnt[1]
                    ):
                        # print(f"Text: {key}, Label: {value}")
                        intersecItem = (key, value)
                        intersecResults.add(intersecItem)

    # set predictions for BERT
    

    # new union #################################################
    for entity in prediction[0]:
        for key, value in entity.items():
            if value not in final_entities:
                if value != "OTHER":
                    unique_entity_BERT = (key, value)
                    if unique_entity_BERT not in final_entities:
                        final_entities.add(unique_entity_BERT)

    for spacyEnt in doc_spacy.ents:
        if spacyEnt.text != "":
            unique_entity_SPACY = (
                spacyEnt.text,
                spacyEnt.label_
            )
            if unique_entity_SPACY not in final_entities:
                final_entities.add(unique_entity_SPACY)

    for fuzzEnt in fuzzyResults:
        unique_entity_FUZZY = (
            fuzzEnt[0],
            fuzzEnt[1],
        )
        if unique_entity_FUZZY not in final_entities:
            final_entities.add(unique_entity_FUZZY)
    #############################################################


    # for entity in prediction[0]:
    #     for key, value in entity.items():
    #         if value not in final_entities:
    #             if value != "OTHER":
    #                 unique_entity = (key, value)  # Use a tuple for uniqueness check

    #                 if unique_entity not in final_entities:
    #                     final_entities.add(unique_entity)

    #             else:
    #                 for ent in doc_spacy.ents:
    #                     if key == ent.text and ent.text != "":
    #                         unique_entity = (
    #                             ent.text,
    #                             ent.label_,
    #                         )  # Use a tuple for uniqueness check
    #                         if unique_entity not in final_entities:
    #                             final_entities.add(unique_entity)

    #                     else:
    #                         for fuzzEnt in fuzzyResults:
    #                             if key == fuzzEnt[0]:
    #                                 unique_entity = (
    #                                     fuzzEnt[0],
    #                                     fuzzEnt[1],
    #                                 )
    #                                 if unique_entity not in final_entities:
    #                                     final_entities.add(unique_entity)



    # print("intersec: ")
    # print(intersecResults)

    # print("union: ")
    # print(final_entities)

    # predOutput = predOutputGenerator(testString, final_entities)
    # print("pred output")
    # print(predOutput)

    # print("true output array")
    # print(trueOutputsArrGenerator(data['annotations']))

    # print("pred output array")
    # print(predOutputsArrGenerator(data['annotations'], final_entities))

    trueOutputsArr = trueOutputsArrGenerator(data['annotations'])
    # predOutputsArr = predOutputsArrGenerator(data['annotations'], final_entities)
    predOutputsArr = predOutputsArrGenerator(data['annotations'], nlp_ner, model)

    # print("TRUE")
    # print(trueOutputsArr)
    # print("PRED")
    # print(predOutputsArr)

    evaluator = Evaluator(trueOutputsArr, predOutputsArr, tags=['LOCATION', 'PERSON', 'ORGANIZATION', 'DATE', 'TIME'])

    # Returns overall metrics and metrics for each tag
    results, results_per_tag = evaluator.evaluate()

    print(results)
    # print(results_per_tag)
