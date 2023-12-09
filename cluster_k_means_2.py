import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import gensim

import json
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
from multiprocessing import freeze_support
from simpletransformers.ner import NERModel, NERArgs
from combined_ner_SBF import namedEntityRecognition

import numpy as np
from sklearn.metrics import silhouette_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import StandardScaler

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
        [
            "තරගයේ කාසියේ වාසිය දිනූ ඉන්දීය කණ්ඩායමේ නායක හර්දික් පාණ්ඩ්‍යා පළමුව පන්දුවට පහරදීමට තීරණය කළේය.\r",
            {"entities": [[24, 30, "LOCATION"], [45, 62, "PERSON"]]},
        ],
        [
            "ඒ අනුව ඉන්දීය කණ්ඩායම විසින් නියමිත පන්දුවාර 20 අවසානයේ කඩුලු 5ක් දැවී ශ්‍රී ලංකාවට ලකණු 229ක දැවැන්ත ඉලක්කයක් ලබා දෙනු ලැබීය.\r",
            {"entities": [[7, 13, "LOCATION"], [71, 83, "LOCATION"]]},
        ],
        [
            "ඉන්දීය පිතිකරු බලකායේ දක්ෂ පිතිකරුවෙකු වන සූර්යකුමාර් යාදව් විශිෂ්ට පිති හරඹයක නිරත වූ අතර පන්දු 51කින් නොදැවී ලකුණු 112ක් රැස්කිරීමට ඔහු සමත්විය.\r",
            {"entities": [[0, 6, "LOCATION"], [42, 59, "PERSON"]]},
        ],
        [
            "සූර්යකුමාර් යාද්ව්ගේ ප්‍රහාරක වේගය 219.60 ක් ලෙස වාර්තා විය.\r",
            {"entities": [[0, 20, "PERSON"]]},
        ],
        [
            "ශුබ්මන් ගිල් ද පන්දු 36කින් වේගවත් ලකණු 46ක් රැස් කළේය.\r",
            {"entities": [[0, 12, "PERSON"]]},
        ],
        [
            "පිළිතුරු ඉනිම ක්‍රීඩා කළ ශ්‍රී ලංකා කණ්ඩායමේ ආරම්භක පිතිකරුවන් වන පැතුම් නිශ්ශංක සහ කුසල් මෙන්ඩිස් තරගයට මනා ආරම්භයක් ලබා දීමට උත්සහ කළත් ඔවුන්ට සාර්ථක වීම ඉන්දීය පන්දු යවන්නන්ගේ බල ඇණියෙන් ඉඩක් නොවීය.\r",
            {
                "entities": [
                    [25, 35, "LOCATION"],
                    [66, 80, "PERSON"],
                    [84, 98, "PERSON"],
                    [156, 162, "LOCATION"],
                ]
            },
        ],
        [
            "ලකුණු 229 ක ඉලක්කයක් හඹා ගිය ශ්‍රී ලංකා කණ්ඩායමේ කිසිදු පිතිකරුවෙකුට ලකුණු 23 සීමාව පසු කිරීමට ඉන්දීය පන්දු යවන්නන්ගෙන් ඉඩක් නොවීය.\r",
            {"entities": [[29, 39, "LOCATION"], [95, 101, "LOCATION"]]},
        ],
        [
            "කුසල් මෙන්ඩිස් සහ නායක දසුන් ශානක ලකුණු 23 බැගින් රැස් කළහ.\r",
            {"entities": [[0, 14, "PERSON"], [23, 33, "PERSON"]]},
        ],
        [
            " දසුන්ගේ එම ලකුණු 23ට 6 පහර 02ක් ද අයත්විය.\r",
            {"entities": [[1, 8, "PERSON"]]},
        ],
        [
            "ඒ සමගම ඉන්දියාවට එරෙහි මෙම තරගාවලිය තුළ ඔහු එල්ල කළ 6 පහර සංඛ්‍යාව 9කි.\r",
            {"entities": [[7, 16, "LOCATION"]]},
        ],
        [
            "කෙසේවෙතත් ශ්‍රී ලංකා කණ්ඩායමේ සියලු ක්‍රීඩකයින් පන්දුවාර 16.4ක දී දැවී ගියේ ලකුණු පුවරුව ලකුණු 137ක් ව තිබියදීය.\r",
            {"entities": [[10, 20, "LOCATION"]]},
        ],
        [
            "තරගයේ වීරයා ලෙස සූර්යකුමාර් යාදව් ද තරගාවලියේ වීරයා ලෙස අක්සාර් පටෙල් ද නම් කෙරිණි.\r",
            {"entities": [[16, 33, "PERSON"], [56, 69, "PERSON"]]},
        ],
        [
            "ඉන්දියාවට එරෙහි වේගවත්ම ප්‍රහාරය දසුන්ගෙන්.\r",
            {"entities": [[0, 9, "LOCATION"], [33, 42, "PERSON"]]},
        ],
    ],
}

Word2VecModel = gensim.models.Word2Vec.load('./Word2Vec/testmodel')

def word2vec_converter(word):
    try:
        word_vector = Word2VecModel.wv[word]
        single_value = np.mean(word_vector)
        return single_value
    except KeyError:
        print(f"Word '{word}' not in vocabulary. Returning a default value.")
        return 0


if __name__ == "__main__":
    freeze_support()

    ########## Temporary commented #########
    # nlp_ner = spacy.load("./SpacyNER/model-best")
    # model = NERModel(
    #     "bert", "./BERT_NER/trained_model", labels=label, args=args, use_cuda=False
    # )
    documents = []
    for annotation in data["annotations"]:
        documents.append(annotation[0])
    # named_entities = [namedEntityRecognition(doc, nlp_ner, model) for doc in documents]

    ########## Temporary commented #########

    named_entities = [
        {
            ("ශ්\u200dරී ලංකා", "LOCATION"),
            ("ඉන්දීය", "LOCATION"),
            ("ලංකා", "LOCATION"),
            ("20/20", "DATE"),
            ("ආසියානු", "LOCATION"),
            ("ඉන්දියාව", "LOCATION"),
        },
        {("රාජ්කොට් සෞරාෂ්ට්\u200dර ක්\u200dරිකට්", "LOCATION")},
        {("ඉන්දීය", "LOCATION"), ("හර්දික් පාණ්ඩ්\u200dයා", "PERSON")},
        {
            ("ඉන්දීය", "LOCATION"),
            ("ශ්\u200dරී ලංකාවට", "LOCATION"),
            ("ලංකාවට", "LOCATION"),
        },
        {
            ("ඉන්දීය", "LOCATION"),
            ("සූර්යකුමාර් යාදව්", "PERSON"),
            ("සූර්යකුමාර්", "PERSON"),
        },
        {
            ("21.19.60", "DATE"),
            ("සූර්යකුමාර්", "PERSON"),
            ("සූර්යකුමාර් යාද්ව්ගේ", "PERSON"),
            ("වාර්තා", "DATE"),
        },
        {("ශුබ්මන් ගිල්", "PERSON")},
        {
            ("ශ්\u200dරී ලංකා", "LOCATION"),
            ("ඉන්දීය", "LOCATION"),
            ("ලංකා", "LOCATION"),
            ("ක්\u200dරීඩා", "LOCATION"),
            ("පිළිතුරු", "LOCATION"),
            ("මෙන්ඩිස්", "ORGANIZATION"),
        },
        {("ශ්\u200dරී ලංකා", "LOCATION"), ("ඉන්දීය", "LOCATION"), ("ලංකා", "LOCATION")},
        {("කුසල් මෙන්ඩිස්", "PERSON"), ("දසුන් ශානක", "PERSON"), ("දසුන්", "PERSON")},
        {("දසුන්ගේ", "PERSON")},
        {("ඉන්දියාවට", "LOCATION")},
        {
            ("ශ්\u200dරී ලංකා", "LOCATION"),
            ("ලංකා", "LOCATION"),
            ("ක්\u200dරීඩකයින්", "LOCATION"),
            ("පුවරුව", "TIME"),
        },
        {
            ("සූර්යකුමාර් යාදව්", "PERSON"),
            ("සූර්යකුමාර්", "PERSON"),
            ("අක්සාර් පටෙල්", "PERSON"),
        },
        {("ඉන්දියාවට", "LOCATION"), ("දසුන්ගෙ", "PERSON"), ("දසුන්ගෙන්", "PERSON")},
    ]

    # Create an empty DataFrame with the desired columns
    columns = ["OBJECT", "PERSON", "LOCATION", "ORGANIZATION", "DATE", "TIME"]
    df = pd.DataFrame(columns=columns)

    # Iterate through the list of dictionaries and populate the DataFrame
    for i, obj_set in enumerate(named_entities):
        # row = {"OBJECT": i + 1}
        row = {"OBJECT": documents[i]}
        if i >= len(df):
            df = df._append(row, ignore_index=True)

            for item, label in obj_set:
                if label not in df.columns:
                    df[label] = ""  # Add the column if it doesn't exist

                vectorValue = word2vec_converter(item)

                existing_data = df.at[i, label]
                if pd.notna(existing_data):
                    # If there's existing data, append the new value with a comma
                    df.at[i, label] = (existing_data + vectorValue)/2
                else:
                    # If there's no existing data, set the new value directly
                    df.at[i, label] = vectorValue

    # Fill NaN values with an empty string
    df = df.fillna(0)

    # Drop duplicate rows based on all columns
    df = df.drop_duplicates()

    print(df)

    ## K-means ##
    selected_columns = ['PERSON', 'LOCATION', 'ORGANIZATION', 'DATE', 'TIME']
    data_for_clustering = df[selected_columns]

    # Standardize the data to have mean=0 and variance=1
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow method graph
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.show()

    # Choose the optimal k (number of clusters)
    optimal_k = 6  

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Print clustered documents with their cluster labels
    for cluster_label in range(optimal_k):
        cluster_docs = df[df['Cluster'] == cluster_label]['OBJECT'].tolist()
        print(f"Cluster {cluster_label + 1} Documents:")
        for doc in cluster_docs:
            print(f"  - {doc}")
        print("\n")

    sns.pairplot(df, hue='Cluster', palette='viridis', diag_kind='kde')
    plt.show()