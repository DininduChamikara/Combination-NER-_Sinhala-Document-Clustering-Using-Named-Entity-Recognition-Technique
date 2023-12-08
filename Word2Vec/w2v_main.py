import gensim
import numpy as np

model = gensim.models.Word2Vec.load('testmodel')

def word2vec_converter(word):
    word_vector = model.wv[word]
    single_value = np.mean(word_vector)  # You can also use np.sum() or other aggregation methods
    return single_value


# word_vector = model.wv["ඉන්දීය"]
# single_value = np.mean(word_vector)  # You can also use np.sum() or other aggregation methods
# print(single_value)

print(word2vec_converter("ඉන්දීය"))
