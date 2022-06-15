import os

import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.metrics import recall_score
import elm_kernel_regression
from scipy.stats import pearsonr
from src.model_learning import DataLoader, CascadedNormalizer

def CCC(y_true, y_pred):
    x_mean = np.nanmean(y_true, dtype="float32")
    y_mean = np.nanmean(y_pred, dtype="float32")
    x_var = 1.0 / (len(y_true) - 1) * np.nansum((y_true - x_mean) ** 2)
    y_var = 1.0 / (len(y_pred) - 1) * np.nansum((y_pred - y_mean) ** 2)
    cov = np.nanmean((y_true - x_mean) * (y_pred - y_mean))
    return round((2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2), 4)


def scoring(classifier_type, pred, true_labels):
    assert classifier_type in ["ELM", "DNN"]
    if classifier_type == "ELM":
        emo_pred = pred
        ccc = CCC(true_labels, emo_pred)
        pearson_r_and_p = pearsonr(true_labels, emo_pred)
        seperate_emotion_scores = []
        for i in range(emo_pred.shape[0]):
            seperate_emotion_scores.append(pearsonr(true_labels[i, :], emo_pred[i, :]))
        return ccc, pearson_r_and_p, seperate_emotion_scores


if __name__ == "__main__":
    dl = DataLoader(train_set="train", test_set="devel", ling_model="acoustic",
                    linguistic_utt="words_compare_llds_110pca_200gmm_fv",
                    acoustic_utt="", utt_functionals="")
    emo_label_path = r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\data\high_info.csv"
    val_label_path = r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\data\two_info.csv"
    print(os.curdir)
    x_train, x_test, y_train, y_test = dl.construct_feature_set(emo_label_path)
    x_train, x_test = CascadedNormalizer(x_train, x_test, "z", "power", "l2", 0.5).normalize()
    model = elm_kernel_regression.ELM(c=4)
    model.fit(x_train, y_train)

    pred_probs = model.predict(x_test)
    pred_score = scoring("ELM", pred_probs, y_test)
    print(pred_score)