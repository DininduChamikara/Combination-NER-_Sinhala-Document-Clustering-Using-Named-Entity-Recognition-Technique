import json
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
from multiprocessing import freeze_support
from simpletransformers.ner import NERModel, NERArgs
from combined_ner_SBF import namedEntityRecognition

import numpy as np
from sklearn.metrics import silhouette_score

label = ["LOCATION", "OTHER", "PERSON", "DATE", "ORGANIZATION", "TIME"]

args = NERArgs()
args.num_train_epochs = 10
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

# with open("./SpacyNER/annotations_v1.0.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

data = {
    "classes": ["LOCATION", "PERSON", "ORGANIZATION", "DATE", "TIME"],
    "annotations": [
        [
            "ආසියානු ක්‍රිකට් ශුර ශ්‍රී ලංකා කණ්ඩායම සහ සත්කාරක ඉන්දීය කණ්ඩායම අතර 20/20 ක්‍රිකට් තරගාවලියේ 3 වැනි සහ අවසන් 20/20 ක්‍රිකට් තරගය ලකුණු 91කින් ජයගනිමින් ඉන්දියාව තරගාවලියේ ජය හිමි කර ගනු ලැබීය.\r",
            {
                "entities": [
                    [0, 7, "LOCATION"],
                    [21, 31, "LOCATION"],
                    [51, 57, "LOCATION"],
                    [154, 162, "LOCATION"],
                ]
            },
        ],
        [
            "රාජ්කොට් සෞරාෂ්ට්‍ර ක්‍රිකට් සංගම් ක්‍රීඩාංගණයේදී තරගය පැවැත්විණි.\r",
            {"entities": [[0, 49, "LOCATION"]]},
        ],
        # [
        #     "තරගයේ කාසියේ වාසිය දිනූ ඉන්දීය කණ්ඩායමේ නායක හර්දික් පාණ්ඩ්‍යා පළමුව පන්දුවට පහරදීමට තීරණය කළේය.\r",
        #     {"entities": [[24, 30, "LOCATION"], [45, 62, "PERSON"]]},
        # ],
        # [
        #     "ඒ අනුව ඉන්දීය කණ්ඩායම විසින් නියමිත පන්දුවාර 20 අවසානයේ කඩුලු 5ක් දැවී ශ්‍රී ලංකාවට ලකණු 229ක දැවැන්ත ඉලක්කයක් ලබා දෙනු ලැබීය.\r",
        #     {"entities": [[7, 13, "LOCATION"], [71, 83, "LOCATION"]]},
        # ],
        # [
        #     "ඉන්දීය පිතිකරු බලකායේ දක්ෂ පිතිකරුවෙකු වන සූර්යකුමාර් යාදව් විශිෂ්ට පිති හරඹයක නිරත වූ අතර පන්දු 51කින් නොදැවී ලකුණු 112ක් රැස්කිරීමට ඔහු සමත්විය.\r",
        #     {"entities": [[0, 6, "LOCATION"], [42, 59, "PERSON"]]},
        # ],
        # [
        #     "සූර්යකුමාර් යාද්ව්ගේ ප්‍රහාරක වේගය 219.60 ක් ලෙස වාර්තා විය.\r",
        #     {"entities": [[0, 20, "PERSON"]]},
        # ],
        # [
        #     "ශුබ්මන් ගිල් ද පන්දු 36කින් වේගවත් ලකණු 46ක් රැස් කළේය.\r",
        #     {"entities": [[0, 12, "PERSON"]]},
        # ],
        # [
        #     "පිළිතුරු ඉනිම ක්‍රීඩා කළ ශ්‍රී ලංකා කණ්ඩායමේ ආරම්භක පිතිකරුවන් වන පැතුම් නිශ්ශංක සහ කුසල් මෙන්ඩිස් තරගයට මනා ආරම්භයක් ලබා දීමට උත්සහ කළත් ඔවුන්ට සාර්ථක වීම ඉන්දීය පන්දු යවන්නන්ගේ බල ඇණියෙන් ඉඩක් නොවීය.\r",
        #     {
        #         "entities": [
        #             [25, 35, "LOCATION"],
        #             [66, 80, "PERSON"],
        #             [84, 98, "PERSON"],
        #             [156, 162, "LOCATION"],
        #         ]
        #     },
        # ],
        # [
        #     "ලකුණු 229 ක ඉලක්කයක් හඹා ගිය ශ්‍රී ලංකා කණ්ඩායමේ කිසිදු පිතිකරුවෙකුට ලකුණු 23 සීමාව පසු කිරීමට ඉන්දීය පන්දු යවන්නන්ගෙන් ඉඩක් නොවීය.\r",
        #     {"entities": [[29, 39, "LOCATION"], [95, 101, "LOCATION"]]},
        # ],
        # [
        #     "කුසල් මෙන්ඩිස් සහ නායක දසුන් ශානක ලකුණු 23 බැගින් රැස් කළහ.\r",
        #     {"entities": [[0, 14, "PERSON"], [23, 33, "PERSON"]]},
        # ],
        # [
        #     " දසුන්ගේ එම ලකුණු 23ට 6 පහර 02ක් ද අයත්විය.\r",
        #     {"entities": [[1, 8, "PERSON"]]},
        # ],
        # [
        #     "ඒ සමගම ඉන්දියාවට එරෙහි මෙම තරගාවලිය තුළ ඔහු එල්ල කළ 6 පහර සංඛ්‍යාව 9කි.\r",
        #     {"entities": [[7, 16, "LOCATION"]]},
        # ],
        # [
        #     "කෙසේවෙතත් ශ්‍රී ලංකා කණ්ඩායමේ සියලු ක්‍රීඩකයින් පන්දුවාර 16.4ක දී දැවී ගියේ ලකුණු පුවරුව ලකුණු 137ක් ව තිබියදීය.\r",
        #     {"entities": [[10, 20, "LOCATION"]]},
        # ],
        # [
        #     "තරගයේ වීරයා ලෙස සූර්යකුමාර් යාදව් ද තරගාවලියේ වීරයා ලෙස අක්සාර් පටෙල් ද නම් කෙරිණි.\r",
        #     {"entities": [[16, 33, "PERSON"], [56, 69, "PERSON"]]},
        # ],
        # [
        #     "ඉන්දියාවට එරෙහි වේගවත්ම ප්‍රහාරය දසුන්ගෙන්.\r",
        #     {"entities": [[0, 9, "LOCATION"], [33, 42, "PERSON"]]},
        # ],
    ],
}


if __name__ == "__main__":
    freeze_support()

    nlp_ner = spacy.load("./SpacyNER/model-best")

    model = NERModel(
        "bert", "./BERT_NER/trained_model", labels=label, args=args, use_cuda=False
    )

    documents = []

    for annotation in data["annotations"]:
        documents.append(annotation[0])

    named_entities = [namedEntityRecognition(doc, nlp_ner, model) for doc in documents]

    print("## NAMED ENTITIES ##")
    print(named_entities)

    # Extracting the first element from each set
    named_entities_lists = [
        [next(iter(s)) for s in sublist] for sublist in named_entities
    ]

    # Combine into one array
    combined_list_NEs = []
    for inner_list in named_entities_lists:
        combined_list_NEs.extend(inner_list)

    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_list_NEs)

    print("## tfidf_matrix ##")
    print(tfidf_matrix)

    # Perform K-Means clustering
    num_clusters = 3  # You can adjust the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    cluster_labels = kmeans.labels_

    # Print the documents and their cluster assignments
    for i, doc in enumerate(documents):
        print(f"Document: {doc}\nCluster: {cluster_labels[i]}\n")
        # if cluster_labels[i] == 3:
        #     print(f"Document: {doc}\nCluster: {cluster_labels[i]}\n")

    # Convert the list to a NumPy array
    combined_list_NEs_array = np.array(combined_list_NEs)

    # Reshape the array to 2D
    combined_list_NEs_2d = combined_list_NEs_array.reshape(-1, 1)

    # Calculate Silhoutte Score
    score = silhouette_score(tfidf_matrix, cluster_labels, metric="euclidean")

    # Print the score
    print("Silhouetter Score: %.3f" % score)
