import os
import wave
import contextlib
import pandas
import pandas as pd
from tqdm import tqdm


def get_length_and_size(audiofile):
    with contextlib.closing(wave.open(audiofile, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)
        size = os.path.getsize(audiofile)
        return length, size


path = r'C:\Users\joren\Documents\Uni\Jaar_4\Scriptie\data\wav'
df = pandas.DataFrame(columns=["Name", "Length", "Size"])
for file in tqdm(os.listdir(path)):
    length, size = get_length_and_size(os.path.join(path, file))
    df.loc[len(df)] = [file, length, size]

df.to_csv("data_length_size", index=False)
