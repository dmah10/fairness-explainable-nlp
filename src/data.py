import os
import numpy as np
import pandas as pd
import json
from datasets import load_dataset, Dataset, Features, Value, ClassLabel
from gender_guesser import detector


CAT_STRINGS = {
    "fashion": "AMAZON_FASHION_5",
    "beauty": "All_Beauty_5",
    "movies": "Movies_and_TV_5",
    "sports": "Sports_and_Outdoors_5",
}


def gecobench_df_to_hf(df):
    df.drop(
        labels=["ground_truth", "sentence_idx", "sentence", "target"],
        inplace=True,
        axis=1,
    )

    d = Dataset.from_pandas(
        df,
        features=Features(
            {
                "text": Value("string"),
                "gender": Value("string"),
                "label": Value("double"),
                "mask": Value("string"),
            }
        ),
        preserve_index=False,
    ).class_encode_column("label")
    return d


def resplit_gecobench(trainset, testset, train_rate=0.8):
    df = pd.concat((trainset, testset))
    indices = df["sentence_idx"].unique()
    train_indices = np.random.choice(
        indices, size=(int(train_rate * len(indices))), replace=False
    )
    test_indices = np.array([i for i in indices if i not in train_indices])

    train_df = df[df["sentence_idx"].isin(train_indices)]
    test_df = df[df["sentence_idx"].isin(test_indices)].sort_values(
        ["sentence_idx", "label"]
    )

    train_dataset = gecobench_df_to_hf(train_df)
    test_dataset = gecobench_df_to_hf(test_df)
    return train_dataset, test_dataset


def load_gecobench(path, return_df=False):
    """
    Return hf dataset containing the sentences and genders.
    """
    lines = []
    with open(path) as f:
        lines = f.read().splitlines()

    line_dicts = [json.loads(line) for line in lines]
    df = pd.DataFrame(line_dicts)
    df["text"] = df.apply(lambda x: " ".join(x["sentence"]), axis=1)
    df["label"] = df["target"]
    df["gender"] = df.apply(lambda x: "male" if x["label"] == 1 else "female", axis=1)
    df["mask"] = df.apply(lambda x: str(x["ground_truth"]), axis=1)

    if return_df:
        return df

    df.drop(
        labels=["ground_truth", "sentence_idx", "sentence", "target"],
        inplace=True,
        axis=1,
    )

    dataset = gecobench_df_to_hf(df)
    return dataset


def genderify(df, n=10_000):
    """
    Duplicates each review in df and prepends random male/female names to each.
    """
    n = min(n, len(df))  # make sure we don't sample more than the population size
    male_names = df[df["gender"] == "male"]["reviewerName"]
    female_names = df[df["gender"] == "female"]["reviewerName"]

    def get_random_name(gender):
        names = male_names if gender == "male" else female_names
        return names.sample(1).item().split(" ")[0]

    split_2_male = df.sample(n=n)  # sample this many
    split_2_female = split_2_male.copy()
    split_2_male["text"] = split_2_male.apply(
        lambda x: f'{get_random_name("male")} wrote: {x["text"]}', axis=1
    )
    split_2_male["gender"] = "male"
    split_2_female["text"] = split_2_female.apply(
        lambda x: f'{get_random_name("female")} wrote: {x["text"]}', axis=1
    )
    split_2_female["gender"] = "female"

    split_2 = pd.concat([split_2_male, split_2_female])
    split_2.drop(labels=["reviewerName"], inplace=True, axis=1)

    return split_2


def load_raw(categories, min_length, max_length, dir, download=True):
    gender_detector = detector.Detector()
    for category in categories:
        name = CAT_STRINGS[category]
        if download:
            download_reviews(name, dir)
        df = get_df(f"{dir.absolute().__str__()}/{name}.json")
        df = df.rename(columns={"reviewText": "text"})

        df = df[
            (min_length < df["text"].str.len()) & (df["text"].str.len() < max_length)
        ]

        df["gender"] = (
            df["reviewerName"]
            .astype("str")
            .apply(lambda x: gender_detector.get_gender(x.split(" ")[0]))
        )

        mfs = df[df["gender"].isin(["male", "female"])]

        mfs.drop(
            labels=[
                "verified",
                "reviewTime",
                "reviewerID",
                "asin",
                "style",
                "unixReviewTime",
                "image",
                "summary",
                "vote",
            ],
            inplace=True,
            axis=1,
        )
        mfs.dropna(axis=0, inplace=True)
        mfs.rename(columns={"overall": "label"}, inplace=True)

        # 0. use as is
        split_0 = mfs.copy()

        # 1. prepend review with first name
        split_1 = mfs.copy()
        split_1["text"] = split_1.apply(
            lambda x: f'{x["reviewerName"].split(" ")[0]} wrote: {x["text"]}', axis=1
        )

        # 2. for each review, prepend random male and female names
        #     takes a long time so have only for a smaller subset now
        split_2 = genderify(mfs)

        split_0.to_csv(f"{dir.absolute().__str__()}/{name}_split_0.csv", index=False)
        split_1.to_csv(f"{dir.absolute().__str__()}/{name}_split_1.csv", index=False)
        split_2.to_csv(f"{dir.absolute().__str__()}/{name}_split_2.csv", index=False)

    return load_processed(
        names=[CAT_STRINGS[category] for category in categories], dir=dir
    )


def load_processed(names, dir):
    splits = [[], [], []]
    for name in names:
        split_0 = pd.read_csv(f"{dir.absolute().__str__()}/{name}_split_0.csv")
        split_1 = pd.read_csv(f"{dir.absolute().__str__()}/{name}_split_1.csv")
        split_2 = pd.read_csv(f"{dir.absolute().__str__()}/{name}_split_2.csv")
        splits[0].append(split_0)
        splits[1].append(split_1)
        splits[2].append(split_2)
    return pd.concat(splits[0]), pd.concat(splits[1]), pd.concat(splits[2])


def convert_to_hf(dfs):
    """
    Takes in a list of dataframes and merges them into a list of hf datasets.
    """
    ds = []
    for df in dfs:
        d = Dataset.from_pandas(
            df,
            features=Features(
                {
                    "text": Value("string"),
                    "__index_level_0__": Value("string"),
                    "gender": Value("string"),
                    "label": Value("double"),
                }
            ),
        ).class_encode_column("label")
        d.remove_columns(["__index_level_0__"])
        ds.append(d)
    return ds


def download_reviews(name, dir):
    os.system(
        f"wget -nc http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/{name}.json.gz -P {dir.absolute().__str__()}"
    )
    os.system(f"yes n | gzip -d {dir.absolute().__str__()}/{name}.json.gz")


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def parse(path):
    g = open(path, "rb")
    for l in g:
        yield json.loads(l)
