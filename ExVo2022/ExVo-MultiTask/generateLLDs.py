import numpy as np
import opensmile
import pandas as pd
import os
import time
from tqdm import tqdm

path = r"C:/Users/user/PycharmProjects/scriptie/ExVo2022/ExVo-MultiTask/data/wav/"

if __name__ == "__main__":
    smileLLD = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    smileLLD_Delta = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas
    )
    files_to_process = os.listdir(path)
    labels = pd.read_csv(r"C:/Users/user/PycharmProjects/scriptie/ExVo2022/ExVo-MultiTask/data/high_info.csv")
    train_files = labels[labels['Split'] == 'Train']['File_ID'].to_numpy()
    test_files = labels[labels['Split'] == 'Test']['File_ID'].to_numpy()
    val_files = labels[labels['Split'] == 'Val']['File_ID'].to_numpy()
    allready_done = np.concatenate((
        os.listdir(r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\Dani_scripts\data\compare_lld_csv\test"),
        os.listdir(r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\Dani_scripts\data\compare_lld_csv\train"),
        os.listdir(r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\Dani_scripts\data\compare_lld_csv\devel")))
    for file in tqdm(files_to_process):
        if f"{file.split('.')[0]}.csv" in allready_done:
            continue
        try:
            d1 = smileLLD.process_file(file, root=path)
        except Exception as e:
            print("Error with file: " + file)
            print(e)
            continue
        d2 = smileLLD_Delta.process_file(file, root=path)
        d3 = pd.concat([d1.reset_index().drop(columns=["start", "end", "file"]),
                        d2.reset_index().drop(columns=["start", "end", "file"])], axis=1)
        d3 = d3.interpolate(method='pad', limit_direction='forward')
        file = file.split(".")[0]
        if f"[{file}]" in train_files:
            d3.to_csv(rf"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask" +
                      rf"\Dani_scripts\data\compare_lld_csv\train\{file}.csv",
                      index=False)
        elif f"[{file}]" in test_files:
            d3.to_csv(rf"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask" +
                      rf"\Dani_scripts\data\compare_lld_csv\test\{file}.csv",
                      index=False)
        elif f"[{file}]" in val_files:
            d3.to_csv(rf"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask" +
                      rf"\Dani_scripts\data\compare_lld_csv\devel\{file}.csv",
                      index=False)
