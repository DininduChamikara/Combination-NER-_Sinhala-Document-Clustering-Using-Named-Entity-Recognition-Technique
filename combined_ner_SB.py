from multiprocessing import freeze_support
from simpletransformers.ner import NERModel, NERArgs
import spacy

label = ["LOCATION", "OTHER", "PERSON", "DATE", "ORGANIZATION", "TIME"]

testString = "2023-2025 ලෝක ටෙස්ට් ශූරතා තරගාවලිය එළඹෙන ජූනි මාසයේ දී ආරම්භ වීමට නියමිත යි. ඒ සඳහා සහභාගි වන එක් එක් කණ්ඩායමට හිමි වන ටෙස්ට් තරග ප්‍රමාණය මේ වන විට ප්‍රකාශයට පත් කර අවසන්. ඒ අනුව මෙවර තරගාවලිය යටතේ ද ශ්‍රී ලංකාවට ටෙස්ට් තරග 12කට ක්‍රීඩා කිරීමේ අවස්ථාව ලැබෙනවා. එයින් ටෙස්ට් තරග 6ක් ශ්‍රී ලංකාවේ ද ඉතිරි ටෙස්ට් තරග 6 විදෙස් රටවල්වල දී ද පැවැත්වීමට නියමිත යි.අවසන් වරට පැවති 2021-2023 ලෝක ටෙස්ට් ශූරතා තරගාවලියේ දී ශ්‍රී ලංකාව සැලකිය යුතු තරම් දස්කම් පෙන්වීමට සමත් වුණා. එනිසා මෙවර ලෝක ටෙස්ට් ශූරතාවේ අවසන් තරගය සඳහා පිවිසීමට ශ්‍රී ලංකාවට තිබෙන අවස්ථාව පිළිබඳව යම් විශ්ලේෂණයක් මෙලෙසින් ඔබ වෙත ගෙන එනවා."

args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

if __name__ == "__main__":
    freeze_support()

    nlp_ner = spacy.load("./SpacyNER/model-best")

    doc_spacy = nlp_ner(testString)

    model = NERModel(
        "bert", "./BERT_NER/trained_model", labels=label, args=args, use_cuda=False
    )

    prediction, model_output = model.predict([testString])

    final_entities = set()

    # print intersec
    print("intersec: ")
    for entity in prediction[0]:
        for key, value in entity.items():
            for ent in doc_spacy.ents:
                if key == ent.text and value == ent.label_:
                    print(f"Text: {key}, Label: {value}")

    # print union
    print("Union: ")
    for entity in prediction[0]:
        for key, value in entity.items():
            if value not in final_entities:
                if value != "OTHER":
                    unique_entity = (key, value)  # Use a tuple for uniqueness check

                    if unique_entity not in final_entities:
                        final_entities.add(unique_entity)
                        print(f"Text: {key}, Label: {value}")
                else:
                    for ent in doc_spacy.ents:
                        if key == ent.text and ent.text != "":
                            unique_entity = (
                                ent.text,
                                ent.label_,
                            )  # Use a tuple for uniqueness check
                            if unique_entity not in final_entities:
                                final_entities.add(unique_entity)
                                print(f"Text: {ent.text}, Label: {ent.label_}")

    print ("successfully prints both union and intersec of Spacy with BERT")
