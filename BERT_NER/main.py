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

print(label)

args = NERArgs()
args.num_train_epochs = 10
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

if __name__ == "__main__":
    freeze_support()

    model = NERModel("bert", "bert-base-cased", labels=label, args=args, use_cuda=False)
    model.train_model(train_data, eval_data=test_data)

    # Define the path where you want to save the model
    save_model_path = 'trained_model'

    # Save the model to the specified directory
    # model.save_model(save_model_path)

    # model.saved_model.save(save_model_path)
    model.model.save_pretrained(save_model_path)
    model.tokenizer.save_pretrained(save_model_path)
    model.config.save_pretrained('trained_model/')

    result, model_outputs, preds_list = model.eval_model(test_data)

    print("Result : ",result)

    # prediction, model_output = model.predict(["ආසියානු ක්‍රිකට් ශුර ශ්‍රී ලංකා කණ්ඩායම සහ සත්කාරක ඉන්දීය කණ්ඩායම"])

    # print("Prediction : ", prediction)




