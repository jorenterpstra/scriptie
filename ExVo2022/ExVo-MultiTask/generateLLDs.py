import opensmile
import pandas as pd
import os
import time
from tqdm import tqdm

path = r"C:/Users/user/PycharmProjects/scriptie/ExVo2022/ExVo-MultiTask/data/wav/"

if __name__ == "__main__":
    t1 = time.time_ns()
    smileLLD = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    smileLLD_Delta = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas
    )
    files_to_process = os.listdir(path)
    for file in tqdm(files_to_process):
        d1 = smileLLD.process_file(file, root=path)
        d2 = smileLLD_Delta.process_file(file, root=path)
        d3 = pd.concat([d1.reset_index().drop(columns=["start", "end", "file"]),
                        d2.reset_index().drop(columns=["start", "end", "file"])], axis=1)
        d3 = d3.interpolate(method='pad', limit_direction='forward')
        d3.to_csv(r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\Dani_scripts\data\compare_lld_csv" +
                  "\\" + file.split('.')[0] + ".csv",
                  index=False)
    t2 = time.time_ns()
    t3 = (t2 - t1) / 1000000000
    print(f"seconds taken: {t3}, time for everthing: {t3 / len(files_to_process) * 59201 / 60 / 60}")