import pandas as pd


def column_to_country(column) -> list:
    countries = {0: "China", 1: "South Africa", 2: "United States", 3: "Venezuela"}
    country_nr = column.idxmax(axis=1)
    truth = []
    for nr in country_nr.iteritems():
        truth.append(countries[int(nr[1])])
    return truth


if __name__ == "__main__":
    original_labels = pd.read_csv("data/data_info.csv")
    emotions = ['Awe', 'Excitement', 'Amusement', 'Awkwardness', 'Fear', 'Horror',
                'Distress', 'Triumph', 'Sadness', 'Surprise']
    emotion_labels = original_labels[["File_ID"] + emotions]
    age_labels = original_labels[["File_ID", "Age"]]
    country_labels = original_labels[["File_ID"]]
    one_hot_countries = pd.get_dummies(original_labels["Country"])
    country_labels = country_labels.join(one_hot_countries)

    final_labels = country_labels.merge(age_labels, on="File_ID").merge(emotion_labels, on="File_ID")
    train_labels = final_labels[final_labels["File_ID"].isin(original_labels[original_labels["Split"]=="Train"]["File_ID"])]
    val_labels = final_labels[final_labels["File_ID"].isin(original_labels[original_labels["Split"]=="Val"]["File_ID"])]
    test_labels = final_labels[final_labels["File_ID"].isin(original_labels[original_labels["Split"]=="Test"]["File_ID"])]
    train_labels.to_csv("data/train_labels.csv", index=False)
    val_labels.to_csv("data/val_labels.csv", index=False)
    test_labels.to_csv("data/test_labels.csv", index=False)