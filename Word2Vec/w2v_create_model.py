import nltk
import gensim
from gensim import corpora, models, similarities
import json
from tqdm import tqdm

with open("../SpacyNER/annotations_v1.0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

corpus = []

for annotation in tqdm(data["annotations"]):
    text = annotation[0]
    corpus.append(text)

tok_corp = [nltk.word_tokenize(sent) for sent in corpus]

model = gensim.models.Word2Vec(tok_corp, min_count=1)

model.save('testmodel')
model = gensim.models.Word2Vec.load('testmodel')

vector = model.wv["ලංකා"]

print(vector)