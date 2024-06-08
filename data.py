import pandas as pd


def getdata():
    data = pd.read_csv("data/P1.csv")
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data = data["Joke"].values
    return list(data)


def main():
    data = getdata()
    with open("data/train.txt", "w") as f:
        for text in data:
            f.write(text + "\n")


if __name__ == "__main__":
    main()