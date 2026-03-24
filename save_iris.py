import os
import pandas as pd
from sklearn.datasets import load_iris


def main():
    iris = load_iris()

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/iris.csv", index=False)

    print("Saved to data/iris.csv")


if __name__ == "__main__":
    main()