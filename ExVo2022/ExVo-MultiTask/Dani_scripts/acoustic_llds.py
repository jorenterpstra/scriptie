import os
import pandas as pd
import pickle
import sys
import numpy as np
from tqdm import tqdm


class AcousticLLDs:
    def __init__(self, llds: str, data: str, header_rows, redundant_cols: int, N: int, delim):
        self.directory = f"data/{llds}_func_csv/"
        self.data = data
        self.llds = llds
        self.header = header_rows
        self.n = N
        assert redundant_cols >= 0
        self.redundant_cols = redundant_cols
        self.delim = delim

    def calculate_acoustic_llds(self):
        ds_folder = f"{self.directory}{self.data}"
        dataset_llds = []
        file_index = 0
        files_ordered = []

        for file in os.listdir(ds_folder):
            if file == ".gitignore":
                continue
            files_ordered.append(file)
        files_ordered.sort()

        for file in tqdm(files_ordered):
            df = pd.read_csv(os.path.join(ds_folder, file), header=self.header, delimiter=self.delim)
            df = df.drop(df.columns[0:self.redundant_cols], axis=1)
            arr = df.values
            # arr = np.float32(arr)       # USE THIS LINE TO decrease float64 to float32
            # if self.redundant_cols > 0:
            #     arr = arr[:, self.redundant_cols:]
            dataset_llds.append(arr)

        with open(f"data/acoustics_pickle/{self.data}_{self.llds}_funcs.pickle", "wb") as f:
            pickle.dump(dataset_llds, f)


if __name__ == "__main__":
    # lldss = "acoustic" # alias for mfcc + rastaplp
    # header_rowss = None  # set to None if there is no header, set to 0 if first row is header
    # redundant_colss = 0  # set to x to skip first x columns
    # delim = ","  # set to "," for mfcc+rasta, or ";" for compare llds

    lldss = "compare"
    header_rowss = 0
    redundant_colss = 2
    delim = ";"


    for ds in ["train", "devel", "test"]:
        wa = AcousticLLDs(lldss, ds, header_rowss, redundant_colss, 0, delim)
        wa.calculate_acoustic_llds()
