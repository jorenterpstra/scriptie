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
        # pearson_r_and_p = pearsonr(np.squeeze(true_labels), np.squeeze(emo_pred))
        seperate_emotion_scores = []
        for i in range(emo_pred.shape[1]):
            seperate_emotion_scores.append(pearsonr(true_labels[:, i], emo_pred[:, i]))
        return ccc, seperate_emotion_scores


class Target_Loader:
    def __init__(self, target_dict, target_paths):
        """
        :param target_dict: dict of target data
        :param target_paths: dict of target paths
        """

        self.target_dict = target_dict
        self.target_paths = target_paths

    def load_target(self, target_type):
        """
        :param target_type: target type
        :return: target data paths and column names
        """
        assert target_type in self.target_dict.keys() and target_type in self.target_paths.keys()
        if target_type == "emo" or target_type == "sentiment":
            return self.target_dict[target_type], self.target_paths[target_type]
        elif target_type == "both":
            return self.target_dict["emo"] + self.target_dict["sentiment"], self.target_paths["both"]


if __name__ == "__main__":
    """
    ELM pipeline
    to use bigger datasets, go to model_learning.py and change the numbers in the __determine_size__ function in the
    DataLoader class, the correct sizes should be there already
    """
    # variables to be changed
    train_set = "train"
    test_set = "devel"
    target_type = ""
    ling_model = "acoustic"
    # which set of fv to use
    linguistic_utt = "words_compare_llds_110pca_200gmm_fv"
    acoustic_utt = ""
    utt_functionals = ""
    gmm_components = 0
    nr_to_remove = 0
    # end of variables to be changed
    # variables to be changed, which are more hardcoded
    paths = {
        "emo": r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\data\high_info.csv",
        "sentiment": r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\data\two_info.csv",
        "both": r"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\data\high_two_info.csv"
    }

    target_dict = {
        "emo": [
            "Awe",
            "Excitement",
            "Amusement",
            "Awkwardness",
            "Fear",
            "Horror",
            "Distress",
            "Triumph",
            "Sadness",
            "Surprise"
        ],

        "sentiment": [
            "Arousal",
            "Valence"
        ]
    }
    # all possible combinations of kernel, power, c
    hyperparameters = {
        "c": [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        "kernel": ["linear", "rbf", "sigmoid"],
        "power": [-1, 0.1, 0.5, 1]
    }


    tl = Target_Loader(target_dict=target_dict, target_paths=paths)
    dl = DataLoader(train_set, test_set, ling_model, linguistic_utt, acoustic_utt, utt_functionals, gmm_components,
                    nr_to_remove)

    for target_type in ["emo", "sentiment"]:
        targets, target_path = tl.load_target(target_type)
        x_train, x_test, y_train, y_test = dl.construct_feature_set(target_path)

        columns = ["c", "kernel", "power", "ccc"] + \
                  ["{}_{}".format(target, "r") for target in targets] + \
                  ["{}_{}".format(target, "p") for target in targets]

        all_scores = pd.DataFrame(columns=columns)

        best_score = 0
        best_params = None
        best_preds = None
        for power in hyperparameters["power"]:
            print("power: ", power)
            x_train, x_test = CascadedNormalizer(x_train, x_test, "z", "power", "l2", power).normalize()
            for c in hyperparameters["c"]:
                print("     c: ", c)
                for kernel in hyperparameters["kernel"]:
                    print("         kernel: ", kernel)
                    model = elm_kernel_regression.ELM(c=c, kernel=kernel)
                    model.fit(x_train, y_train)

                    pred_probs = preprocessing.normalize(model.predict(x_test), norm='l2', axis=1)
                    ccc, scores = scoring("ELM", pred_probs, y_test)
                    all_scores.loc[len(all_scores)] = [c, kernel, power, ccc] + \
                                                      [score[0] for score in scores] + \
                                                      [score[1] for score in scores]

                    if ccc > best_score:
                        best_score = ccc
                        best_params = [c, kernel, power]
                        best_preds = pred_probs

        best_preds = pd.DataFrame(best_preds)
        best_preds.columns = targets
        best_preds.to_csv(rf"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\data\best_preds_{target_type}.csv",
                          index=False)
        best_scores = all_scores.loc[all_scores["ccc"] == best_score]
        best_scores.to_csv(rf"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\data\best_scores_{target_type}.csv",
                           index=False)
        all_scores.to_csv(rf"C:\Users\user\PycharmProjects\scriptie\ExVo2022\ExVo-MultiTask\data\all_scores_{target_type}.csv",
                          index=False)
        print("--------scores--------")
        print(f"With hyperparameters:  c: {best_params[0]}, kernel: {best_params[1]}, and power: {best_params[2]}")
        print(f"CCC: {best_score}")
        for target in targets:
            r = all_scores[(all_scores['ccc'] == best_score) & (all_scores['c'] == best_params[0]) &
                           (all_scores['kernel'] == best_params[1]) & (all_scores['power'] == best_params[2])][
                target + '_r']
            p = all_scores[(all_scores['ccc'] == best_score) & (all_scores['c'] == best_params[0]) &
                           (all_scores['kernel'] == best_params[1]) & (all_scores['power'] == best_params[2])][
                target + '_p']
            print(
                f"{target:11} r-value: {r.values[0]:.4f} p-value: {p.values[0]:.5f}"
            )
        pass
