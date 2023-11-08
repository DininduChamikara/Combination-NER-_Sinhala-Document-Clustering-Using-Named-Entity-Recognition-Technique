import json
import spacy

from spacy.tokens import DocBin
from tqdm import tqdm

with open("./annotations_v1.0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

nlp = spacy.blank("si")  # load a new spacy model

# New Test 2023/11/05

# Test end

doc_bin = DocBin()

from tqdm.gui import tqdm
from spacy.util import filter_spans

for training_example in tqdm(data["annotations"]):
    text = training_example[0]
    labels = training_example[1]["entities"]
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entry")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy")