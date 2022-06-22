import os
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    wav_path = os.path.join("data", "wav")
    base_path = r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask"
    target_path = os.path.join("Dani_scripts", "data", "audio_wav")

    info = pd.read_csv(os.path.join("Dani_scripts", "data", "labels_csv", "high_info.csv"))

    partitions = {
        "train": info[info["Split"] == "Train"]["File_ID"],
        "val": info[info["Split"] == "Val"]["File_ID"],
        "test": info[info["Split"] == "Test"]["File_ID"]
    }

    for file in tqdm(os.listdir(os.path.join(base_path, wav_path))):
        if os.path.isfile(os.path.join(base_path, wav_path, file)):
            if partitions["train"].str.contains(file.split(".")[0]).any():
                os.replace(os.path.join(base_path, wav_path, file), os.path.join(base_path, target_path, "train", file))
            elif partitions["val"].str.contains(file.split(".")[0]).any():
                os.replace(os.path.join(base_path, wav_path, file), os.path.join(base_path, target_path, "devel", file))
            elif partitions["test"].str.contains(file.split(".")[0]).any():
                os.replace(os.path.join(base_path, wav_path, file), os.path.join(base_path, target_path, "test", file))
    pass
