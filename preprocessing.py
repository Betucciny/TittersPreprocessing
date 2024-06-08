import keras
import tensorflow as tf
import keras_nlp
import numpy as np
import pandas as pd


def getdata():
    data = pd.read_csv("data/P1.csv")
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data = data["Joke"].values
    return list(data)


def main():
    features = getdata()
    llama_lm = keras_nlp.models.Llama3CausalLM.from_preset("llama3_8b_en", dtype="bfloat16")
    llama_lm.fit(x=features, batch_size=20)
    llama_lm.save("models/llama3_8b_en")
    
    
if __name__ == "__main__":
    main()
