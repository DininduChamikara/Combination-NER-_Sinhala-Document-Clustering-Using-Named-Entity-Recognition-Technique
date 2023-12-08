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

# data = {
#     "classes": ["LOCATION", "PERSON", "ORGANIZATION", "DATE", "TIME"],
#     "annotations": [
#         [
#             "ආසියානු ක්‍රිකට් ශුර ශ්‍රී ලංකා කණ්ඩායම සහ සත්කාරක ඉන්දීය කණ්ඩායම අතර 20/20 ක්‍රිකට් තරගාවලියේ 3 වැනි සහ අවසන් 20/20 ක්‍රිකට් තරගය ලකුණු 91කින් ජයගනිමින් ඉන්දියාව තරගාවලියේ ජය හිමි කර ගනු ලැබීය.\r",
#             {
#                 "entities": [
#                     [0, 7, "LOCATION"],
#                     [21, 31, "LOCATION"],
#                     [51, 57, "LOCATION"],
#                     [154, 162, "LOCATION"],
#                 ]
#             },
#         ],
#         [
#             "රාජ්කොට් සෞරාෂ්ට්‍ර ක්‍රිකට් සංගම් ක්‍රීඩාංගණයේදී තරගය පැවැත්විණි.\r",
#             {"entities": [[0, 49, "LOCATION"]]},
#         ],
#         [
#             "තරගයේ කාසියේ වාසිය දිනූ ඉන්දීය කණ්ඩායමේ නායක හර්දික් පාණ්ඩ්‍යා පළමුව පන්දුවට පහරදීමට තීරණය කළේය.\r",
#             {"entities": [[24, 30, "LOCATION"], [45, 62, "PERSON"]]},
#         ],
#         [
#             "ඒ අනුව ඉන්දීය කණ්ඩායම විසින් නියමිත පන්දුවාර 20 අවසානයේ කඩුලු 5ක් දැවී ශ්‍රී ලංකාවට ලකණු 229ක දැවැන්ත ඉලක්කයක් ලබා දෙනු ලැබීය.\r",
#             {"entities": [[7, 13, "LOCATION"], [71, 83, "LOCATION"]]},
#         ],
#         [
#             "ඉන්දීය පිතිකරු බලකායේ දක්ෂ පිතිකරුවෙකු වන සූර්යකුමාර් යාදව් විශිෂ්ට පිති හරඹයක නිරත වූ අතර පන්දු 51කින් නොදැවී ලකුණු 112ක් රැස්කිරීමට ඔහු සමත්විය.\r",
#             {"entities": [[0, 6, "LOCATION"], [42, 59, "PERSON"]]},
#         ],
#         [
#             "සූර්යකුමාර් යාද්ව්ගේ ප්‍රහාරක වේගය 219.60 ක් ලෙස වාර්තා විය.\r",
#             {"entities": [[0, 20, "PERSON"]]},
#         ],
#         [
#             "ශුබ්මන් ගිල් ද පන්දු 36කින් වේගවත් ලකණු 46ක් රැස් කළේය.\r",
#             {"entities": [[0, 12, "PERSON"]]},
#         ],
#         [
#             "පිළිතුරු ඉනිම ක්‍රීඩා කළ ශ්‍රී ලංකා කණ්ඩායමේ ආරම්භක පිතිකරුවන් වන පැතුම් නිශ්ශංක සහ කුසල් මෙන්ඩිස් තරගයට මනා ආරම්භයක් ලබා දීමට උත්සහ කළත් ඔවුන්ට සාර්ථක වීම ඉන්දීය පන්දු යවන්නන්ගේ බල ඇණියෙන් ඉඩක් නොවීය.\r",
#             {
#                 "entities": [
#                     [25, 35, "LOCATION"],
#                     [66, 80, "PERSON"],
#                     [84, 98, "PERSON"],
#                     [156, 162, "LOCATION"],
#                 ]
#             },
#         ],
#         [
#             "ලකුණු 229 ක ඉලක්කයක් හඹා ගිය ශ්‍රී ලංකා කණ්ඩායමේ කිසිදු පිතිකරුවෙකුට ලකුණු 23 සීමාව පසු කිරීමට ඉන්දීය පන්දු යවන්නන්ගෙන් ඉඩක් නොවීය.\r",
#             {"entities": [[29, 39, "LOCATION"], [95, 101, "LOCATION"]]},
#         ],
#         [
#             "කුසල් මෙන්ඩිස් සහ නායක දසුන් ශානක ලකුණු 23 බැගින් රැස් කළහ.\r",
#             {"entities": [[0, 14, "PERSON"], [23, 33, "PERSON"]]},
#         ],
#         [
#             " දසුන්ගේ එම ලකුණු 23ට 6 පහර 02ක් ද අයත්විය.\r",
#             {"entities": [[1, 8, "PERSON"]]},
#         ],
#         [
#             "ඒ සමගම ඉන්දියාවට එරෙහි මෙම තරගාවලිය තුළ ඔහු එල්ල කළ 6 පහර සංඛ්‍යාව 9කි.\r",
#             {"entities": [[7, 16, "LOCATION"]]},
#         ],
#         [
#             "කෙසේවෙතත් ශ්‍රී ලංකා කණ්ඩායමේ සියලු ක්‍රීඩකයින් පන්දුවාර 16.4ක දී දැවී ගියේ ලකුණු පුවරුව ලකුණු 137ක් ව තිබියදීය.\r",
#             {"entities": [[10, 20, "LOCATION"]]},
#         ],
#         [
#             "තරගයේ වීරයා ලෙස සූර්යකුමාර් යාදව් ද තරගාවලියේ වීරයා ලෙස අක්සාර් පටෙල් ද නම් කෙරිණි.\r",
#             {"entities": [[16, 33, "PERSON"], [56, 69, "PERSON"]]},
#         ],
#         [
#             "ඉන්දියාවට එරෙහි වේගවත්ම ප්‍රහාරය දසුන්ගෙන්.\r",
#             {"entities": [[0, 9, "LOCATION"], [33, 42, "PERSON"]]},
#         ],
#         [
#             "තරගාවලිය පිළිබඳ අදහස් දක්වමින් ශ්‍රී ලංකා නායක දසුන් ශානක කියා සිටියේ “මෙහෙට එන්න කලින් මම ටොප් ෆෝම් එකේ සිටියේ නැහැ. මම ක්‍රීඩා කළ ආකාරය ගැන ගොඩක් සතුටුයි. කණ්ඩායමේ ක්‍රීඩකයින් තරගාවලියට ක්‍රීඩා කළ ආකාරය බැලුහාම, ධනාත්මක කරුණු රාශියක් තිබුණා. ක්‍රීඩකයෝ ගොඩක් දියුණු වෙලා. ඔවුන් හොඳින් ක්‍රීඩා කරනවා දැකීම සතුටක්.”\r",
#             {"entities": [[31, 41, "LOCATION"], [47, 57, "PERSON"]]},
#         ],
#     ],
# }

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

testingDataPortionDivider = 50  # 20% => 100/5 => 5


def trueOutputsArrGenerator(annotations):
    trueOutputs = []
    testData = annotations[0 : int(len(annotations) / testingDataPortionDivider)]
    for annot in testData:
        trueOutputs.append(trueOutputGenerator(annot))
    return trueOutputs


def predOutputsArrGenerator(annotations, spacyNER, bertNER):
    predOutputs = []
    testData = annotations[0 : int(len(annotations) / testingDataPortionDivider)]
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


def namedEntityRecognition(sentence, spacyNER, bertNER):
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
            unique_entity_SPACY = (spacyEnt.text, spacyEnt.label_)
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

    trueOutputsArr = trueOutputsArrGenerator(data["annotations"])
    predOutputsArr = predOutputsArrGenerator(data["annotations"], nlp_ner, model)

    evaluator = Evaluator(
        trueOutputsArr,
        predOutputsArr,
        tags=["LOCATION", "PERSON", "ORGANIZATION", "DATE", "TIME"],
    )

    # Returns overall metrics and metrics for each tag
    results, results_per_tag = evaluator.evaluate()

    print(results)
    print(results_per_tag)
