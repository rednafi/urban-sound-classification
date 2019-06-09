import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_2d import chunker, load_audio_file, input_to_target
from tensorflow.keras.models import load_model
import glob


batch_size = 16


def test_audio(test_files, list_labels):

    model = load_model("./models/model_2d.h5")
    bag = 5
    array_preds = 0

    for i in tqdm(range(bag)):

        list_preds = []

        for batch_files in tqdm(
            chunker(test_files, size=batch_size), total=len(test_files) // batch_size
        ):
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:, :, :, np.newaxis]
            preds = model.predict(batch_data).tolist()
            list_preds += preds

        array_preds += np.array(list_preds) / bag

    list_labels = np.array(list_labels)

    top_5 = list_labels[np.argsort(-array_preds, axis=1)[:, :5]]
    pred_labels = [" ".join(list(x)) for x in top_5]

    df = pd.DataFrame(test_files, columns=["file_name"])
    df["label"] = pred_labels
    df["file_name"] = df["file_name"].apply(lambda x: x.split("/")[-1])

    df.to_csv("./results/pred_2d.csv", index=False)
    return df


base_data_path = "./data/"
test_path = base_data_path + "test/Test/*.wav"
test_files = glob.glob(test_path)[0:3]

train_file_to_label = input_to_target()
list_labels = train_file_to_label["Class"].unique().tolist()
test_audio(test_files, list_labels)
