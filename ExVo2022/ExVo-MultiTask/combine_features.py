import pandas as pd
from os import listdir
path = "C:\\Users\\joren\\Documents\\Uni\\Jaar_4\\Scriptie\\data\\feats\\"

files = listdir(path + "ComPeGe\\")
for i, file in enumerate(listdir(path + "ComParE\\")):
    if i % 100 == 0:
        print(i)
    if file in files:
        continue
    compare_feat = pd.read_csv(path + "ComParE\\" + file, sep=",")
    egemaps_feat = pd.read_csv(path + "eGeMAPS\\" + file, sep=",")
    combined_feat = compare_feat.merge(egemaps_feat)
    combined_feat.to_csv(path + "ComPeGe\\" + file)