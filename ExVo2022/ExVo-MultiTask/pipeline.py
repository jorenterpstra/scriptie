import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn import preprocessing
from scipy.stats import hmean

from dataloader import Dataloader
from Dani_scripts.src import model_learning

import pandas as pd
from elm_kernel_regression import ELM
from models import MultiTask
from utils import Processing, EvalMetrics


def scoring(classifier_type, pred, true_labels):
    assert classifier_type in ["ELM", "DNN"]
    if classifier_type == "ELM":
        emo_pred = pred[:, :10]
        age_pred = pred[:, 10:11]
        country_pred = np.argmax(pred[:, 11:], axis=1)
        ccc = EvalMetrics.CCC(true_labels[:, :10], emo_pred)
        mae = EvalMetrics.MAE(true_labels[:, 10:11], age_pred)
        uar = EvalMetrics.UAR(true_labels[:, 11:], country_pred)
        return ccc, mae, uar, hmean([ccc, 1 / mae, uar])


def main(feature_to_test, fisher_vector_path="", classifier_type="ELM"):
    train_val_test_x, high, age, country, feat_dimensions, test_filename_group = Dataloader.load(feature_to_test,
                                                                                                 feature_to_test)
    scaler = StandardScaler()
    one_hot_encoder = OneHotEncoder(sparse=False)
    (train_x, val_x, test_x), emo_y, age_y, country_y = Processing.normalise(scaler, train_val_test_x, high, age,
                                                                             country)
    for i, country_set in enumerate(country_y):
        country_y[i] = one_hot_encoder.fit_transform(country_set)

    train_y = np.concatenate([emo_y[0].to_numpy(),
                              age_y[0].to_numpy(),
                              country_y[0]], axis=1)

    val_y = np.concatenate([emo_y[1].to_numpy(),
                            age_y[1].to_numpy(),
                            np.reshape(np.argmax(country_y[1], axis=1), (-1, 1))], axis=1)

    classifier = ELM(c=1, weighted=False, kernel='linear', deg=3, is_classification=False) \
        if classifier_type == "ELM" else MultiTask(feat_dimensions)  # TODO fix that multitask actually works
    classifier.fit(train_x[:1000, :], train_y[:1000, :])
    pred = classifier.predict(val_x[:5000, :])
    ccc, mae, uar, h_mean = scoring(classifier_type, pred, val_y[:5000, :])
    print(f"CCC: {ccc}")
    print(f"MAE: {mae}")
    print(f"UAR: {uar}")
    print(f"Harmonic Mean: {h_mean}")
    pred_frame = pd.DataFrame(pred)
    pass


if __name__ == "__main__":
    main("eGeMAPS")
