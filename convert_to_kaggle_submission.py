import argparse
import os

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


def main(result_path):
    result = pd.read_csv(result_path, sep="\t")
    result["label"] = result["prediction"].map(lambda x: label2idx[x])
    kaggle_output_filename = os.path.basename(result)
    result[["comments", "label"]].to_csv(
        f"{kaggle_dir}/{kaggle_output_filename}.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-path", help="Path to a model prediction result", required=True
    )
    args = parser.parse_args()

    main(args.result_path)
