import spacy
from spacy.scorer import Scorer
from spacy.training import Example

examples = [
    ('රාජ්කොට් සෞරාෂ්ට්‍ර ක්‍රිකට් සංගම් ක්‍රීඩාංගණයේදී තරගය පැවැත්විණි.',
     {(0, 49, "LOCATION")}),
    ('තරගයේ කාසියේ වාසිය දිනූ ඉන්දීය කණ්ඩායමේ නායක හර්දික් පාණ්ඩ්‍යා පළමුව පන්දුවට පහරදීමට තීරණය කළේය.',
     {(24, 30, "LOCATION"), (45, 62, "PERSON")})
]

def evaluate(ner_model, examples):
    scorer = Scorer()
    example = []
    for input_, annot in examples:
        pred = ner_model(input_)
        # print(pred, annot)
        temp = Example.from_dict(pred, dict.fromkeys(annot))
        example.append(temp)
    scores = scorer.score(example)
    return scores

ner_model = spacy.load("./model-best")
results = evaluate(ner_model, examples)
# print(results)

print("Accuracy: ", results['token_acc'])
print("Precision: ", results['token_p'])
print("Recall: ", results['token_r'])
print("F1: ", results['token_f'])
