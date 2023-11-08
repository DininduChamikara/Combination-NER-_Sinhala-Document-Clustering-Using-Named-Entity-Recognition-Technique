from multiprocessing import freeze_support
from simpletransformers.ner import NERModel, NERArgs

label = ['LOCATION', 'OTHER', 'PERSON', 'DATE', 'ORGANIZATION', 'TIME'] 

args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

if __name__ == "__main__":
    freeze_support()

    model = NERModel("bert", './trained_model', labels=label, args=args, use_cuda=False)

    prediction, model_output = model.predict(["2023-2025 ලෝක ටෙස්ට් ශූරතා තරගාවලිය එළඹෙන ජූනි මාසයේ දී ආරම්භ වීමට නියමිත යි. ඒ සඳහා සහභාගි වන එක් එක් කණ්ඩායමට හිමි වන ටෙස්ට් තරග ප්‍රමාණය මේ වන විට ප්‍රකාශයට පත් කර අවසන්. ඒ අනුව මෙවර තරගාවලිය යටතේ ද ශ්‍රී ලංකාවට ටෙස්ට් තරග 12කට ක්‍රීඩා කිරීමේ අවස්ථාව ලැබෙනවා. එයින් ටෙස්ට් තරග 6ක් ශ්‍රී ලංකාවේ ද ඉතිරි ටෙස්ට් තරග 6 විදෙස් රටවල්වල දී ද පැවැත්වීමට නියමිත යි.අවසන් වරට පැවති 2021-2023 ලෝක ටෙස්ට් ශූරතා තරගාවලියේ දී ශ්‍රී ලංකාව සැලකිය යුතු තරම් දස්කම් පෙන්වීමට සමත් වුණා. එනිසා මෙවර ලෝක ටෙස්ට් ශූරතාවේ අවසන් තරගය සඳහා පිවිසීමට ශ්‍රී ලංකාවට තිබෙන අවස්ථාව පිළිබඳව යම් විශ්ලේෂණයක් මෙලෙසින් ඔබ වෙත ගෙන එනවා."])

    print("Prediction : ", prediction)