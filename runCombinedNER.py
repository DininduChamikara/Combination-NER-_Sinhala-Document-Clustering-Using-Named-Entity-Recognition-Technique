import spacy
from multiprocessing import freeze_support
from simpletransformers.ner import NERModel, NERArgs
from combined_ner_SBF import namedEntityRecognition

label = ["LOCATION", "OTHER", "PERSON", "DATE", "ORGANIZATION", "TIME"]

args = NERArgs()
args.num_train_epochs = 10
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

if __name__ == "__main__":
    freeze_support()

    nlp_ner = spacy.load("./SpacyNER/model-best")

    model = NERModel(
        "bert", "./BERT_NER/trained_model", labels=label, args=args, use_cuda=False
    )

    testStr = "ගාඩියන් පුවත්පතට අනුව, 2019 වසරේදී විලියම් හැරීගේ ලන්ඩන් නිවසේ දී හැරීට පැවසූ ඇතැම් කරුණු හේතූවෙන් මෙම ආරවුල ඇතිවී තිබෙන බව කියවේ.\r"

    result = namedEntityRecognition(testStr, nlp_ner, model)

    print(result)
