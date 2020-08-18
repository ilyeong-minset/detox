import argparse

import pandas as pd

from utils import makedirs

result_dir = "results"
kaggle_dir = "kaggle"
makedirs(kaggle_dir)

label2idx = {
    "none": 0,
    "offensive": 1,
    "hate": 2,
}


def main(model_name):
    result_path = f"{result_dir}/{model_name}.predict"
    result = pd.read_csv(result_path, sep="\t")
    result["label"] = result["prediction"].map(lambda x: label2idx[x])
    result[["comments", "label"]].to_csv(f"{kaggle_dir}/{model_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="saved model name", required=True)
    args = parser.parse_args()

    main(args.model_name)
