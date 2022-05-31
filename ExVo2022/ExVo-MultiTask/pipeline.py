import numpy as np
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

    train_y = np.concatenate([emo_y[0].to_numpy(),
                              age_y[0].to_numpy(),
                              country_y[0]], axis=1)

    val_y = np.concatenate([emo_y[1].to_numpy(),
                            age_y[1].to_numpy(),
                            np.reshape(np.argmax(country_y[1], axis=1), (-1, 1))], axis=1)

    classifier = ELM(c=1, weighted=False, kernel='linear', deg=3, is_classification=False) \
        if ELM_or_DNN == "ELM" else MultiTask(feat_dimensions)  # TODO fix that multitask actually works
    classifier.fit(train_x[:10000, :], train_y[:10000, :])
    pred = classifier.predict(val_x[:5001, :])
    emo_pred = pred[:, :10]
    age_pred = pred[:, 10:11]
    country_pred = np.argmax(pred[:, 11:], axis=1)
    ccc = EvalMetrics.CCC(val_y[:5001, :10], emo_pred)
    mae = EvalMetrics.MAE(val_y[:5001, 10:11], age_pred)
    uar = EvalMetrics.UAR(val_y[:5001, 11:], country_pred)
    print(f"CCC: {ccc}")
    print(f"MAE: {mae}")
    print(f"UAR: {uar}")
    print(f"Harmonic Mean: {hmean([ccc, 1/mae, uar])}")
    pred_frame = pd.DataFrame(pred)
    pred_frame.to_csv("data/elm_regression_preds")
