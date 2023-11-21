from multiprocessing import freeze_support
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from simpletransformers.ner import NERModel, NERArgs

data = pd.read_csv("./ner_dataset_v3.0.csv", encoding="utf-8")

data["Sentence #"] = LabelEncoder().fit_transform(data["Sentence #"])
data.rename(
    columns={"Sentence #": "sentence_id", "Word": "words", "Tag": "labels"},
    inplace=True,
)
data["labels"] = data["labels"].str.upper()

X = data[["sentence_id", "words"]]
Y = data["labels"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# building up train data and test data
train_data = pd.DataFrame(
    {
        "sentence_id": x_train["sentence_id"],
        "words": x_train["words"],
        "labels": y_train,
    }
)
test_data = pd.DataFrame(
    {"sentence_id": x_test["sentence_id"], "words": x_test["words"], "labels": y_test}
)

label = data["labels"].unique().tolist()

args = NERArgs()
args.num_train_epochs = 10
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

if __name__ == "__main__":
    freeze_support()

    model = NERModel("bert", './trained_model', labels=label, args=args, use_cuda=False)
    result, model_outputs, preds_list = model.eval_model(test_data)

    print("Result : ",result)