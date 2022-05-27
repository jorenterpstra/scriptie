from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn import preprocessing
from scipy.stats import hmean

from dataloader import Dataloader
import pandas as pd
from elm_kernel_regression import ELM
from models import MultiTask
from utils import Processing, EvalMetrics

feature_to_test = "eGeMAPS"
data_storage = feature_to_test
ELM_or_DNN = "ELM"

if __name__ == "__main__":
    X, high, age, country, feat_dimensions, test_filename_group = Dataloader.load(feature_to_test, data_storage)
    scaler = StandardScaler()
    oneHotEncoder = OneHotEncoder(sparse=False)
    (train_x, val_x, test_x), emo_y, age_y, country_y = Processing.normalise(scaler, X, high, age, country)
    for i, country_set in enumerate(country_y):
        country_y[i] = oneHotEncoder.fit_transform( country_set)
        # TODO make programs use this set instead of saved one for better congruence
    train_y = pd.read_csv("data/train_labels.csv")
    val_y = pd.read_csv("data/val_labels.csv")

    classifier = ELM(c=1, weighted=False, kernel='linear', deg=3, is_classification=False) \
        if ELM_or_DNN == "ELM" else MultiTask(feat_dimensions)  # TODO fix that multitask actually works

    classifier.fit(train_x.head(100), train_y.drop("File_ID", axis=1).head(100))
    pred = classifier.predict(val_x.head(101))
    print(pred)
    print(f"CCC: ")
    pred_frame = pd.DataFrame(pred)
    pred_frame.to_csv("data/elm_regression_preds")
