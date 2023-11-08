import json

with open("./dictionary.json", "r", encoding="utf-8") as f:
    dictionary = json.load(f)


output_file = "dict_sinhala.json"

with open("./annotations_v1.0.json", "r", encoding="utf-8") as f:
    input_annotations = json.load(f)["annotations"]

resultObject = {
    "LOCATION": [],
    "PERSON": [],
    "ORGANIZATION": [],
    "DATE": [],
    "TIME": [],
}

for annotation in input_annotations:
    sentence = annotation[0]
    for entity in annotation[1]["entities"]:
        if entity[2] == "LOCATION":
            resultObject["LOCATION"].append(sentence[entity[0] : entity[1]])
        elif entity[2] == "PERSON":
            resultObject["PERSON"].append(sentence[entity[0] : entity[1]])
        elif entity[2] == "ORGANIZATION":
            resultObject["ORGANIZATION"].append(sentence[entity[0] : entity[1]])
        elif entity[2] == "DATE":
            resultObject["DATE"].append(sentence[entity[0] : entity[1]])
        elif entity[2] == "TIME":
            resultObject["TIME"].append(sentence[entity[0] : entity[1]])

# print(resultObject)

# Write the result to the JSON file
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(resultObject, json_file, ensure_ascii=False, indent=4)
