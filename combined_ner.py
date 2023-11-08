#############
# for Spacy #
#############

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from spacy import displacy

nlp_ner = spacy.load("./SpacyNER/model-best")

testString = "2023-2025 ලෝක ටෙස්ට් ශූරතා තරගාවලිය එළඹෙන ජූනි මාසයේ දී ආරම්භ වීමට නියමිත යි. ඒ සඳහා සහභාගි වන එක් එක් කණ්ඩායමට හිමි වන ටෙස්ට් තරග ප්‍රමාණය මේ වන විට ප්‍රකාශයට පත් කර අවසන්. ඒ අනුව මෙවර තරගාවලිය යටතේ ද ශ්‍රී ලංකාවට ටෙස්ට් තරග 12කට ක්‍රීඩා කිරීමේ අවස්ථාව ලැබෙනවා. එයින් ටෙස්ට් තරග 6ක් ශ්‍රී ලංකාවේ ද ඉතිරි ටෙස්ට් තරග 6 විදෙස් රටවල්වල දී ද පැවැත්වීමට නියමිත යි.අවසන් වරට පැවති 2021-2023 ලෝක ටෙස්ට් ශූරතා තරගාවලියේ දී ශ්‍රී ලංකාව සැලකිය යුතු තරම් දස්කම් පෙන්වීමට සමත් වුණා. එනිසා මෙවර ලෝක ටෙස්ට් ශූරතාවේ අවසන් තරගය සඳහා පිවිසීමට ශ්‍රී ලංකාවට තිබෙන අවස්ථාව පිළිබඳව යම් විශ්ලේෂණයක් මෙලෙසින් ඔබ වෙත ගෙන එනවා."

doc1 = nlp_ner(testString)

colors = {
    "LOCATION": "#F67DE3",
    "PERSON": "#7DF609",
    "ORGANIZATION": "#A6E22D",
    "DATE": "#FFFF00",
    "TIME": "#800000",
}
options = {"colors": colors}

print(doc1.ents)

# try:
#     displacy.serve(
#         doc1, style="ent", options=options, page=True, host="localhost", port=5000
#     )
# except:
#     print("error")


############
# For BERT #
############

from multiprocessing import freeze_support
from simpletransformers.ner import NERModel, NERArgs

label = ["LOCATION", "OTHER", "PERSON", "DATE", "ORGANIZATION", "TIME"]

args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

if __name__ == "__main__":
    freeze_support()

    model = NERModel(
        "bert", "./BERT_NER/trained_model", labels=label, args=args, use_cuda=False
    )

    prediction, model_output = model.predict([testString])

    print("Prediction : ", prediction)


####################
# For FuzzyMatcher #
####################

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json

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


input_sentence = testString

# stop words remove
result = input_sentence.split()
final = []

for i in result:
    if i not in stopwords:
        final.append(i)
        final.append(" ")
stop_removed = "".join(final)

tokens = stop_removed.split()

dictionary = dictionary

for token in tokens:
    fuzzy_ratio = getFuzzyRatio(token=token, confidence_levels=True)
    similarity_score = getFuzzySimilarity(
        token=token, dictionary=dictionary, min_ratio=fuzzy_ratio
    )
    if not similarity_score == None:
        print("'" + token + "' : " + "[" + similarity_score[2] + "]")
