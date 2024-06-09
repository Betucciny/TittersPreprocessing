import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from itertools import product


def setup():
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    torch_dtype=torch.bfloat16, device_map="auto")
    return pipe


def generate(inputs):
    pipe, gender, nationality, age = inputs
    input_text = f"Write me  a daily tweet intended to make me laugh, it could be anything like you would post on sotial media. Focus on witty observations or humorous takes on everyday situations. The tweet should be concise, clever, and include a punchline at the end. Avoid using emojis, rely solely on words for humor. You don't have to explain the joke I'll get it. Thanks!"
    messages = [
        {"role": "system", "content": f"Hi there I'm a {gender} {nationality}, I'm {
            age} years old and I'm here to help you with your daily tweet. I'll do my best to make you laugh!"},
        {"role": "user", "content": input_text}
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=500, do_sample=True,
                   temperature=0.7, top_k=50, top_p=0.95)
    tweet = outputs[0]["generated_text"].split("<|assistant|>")[1].strip()
    return {"age": age, "nationality": nationality, "gender": gender, "tweet": tweet}


def create_csv(tweets):
    df = pd.DataFrame(tweets)
    df.to_csv("tweets.csv", index=False)


def main():
    pipe = setup()
    genders = ["male", "female", "non-binary"]
    nationalities = ["American", "British",
                     "Canadian", "Australian", "Indian", "Mexican"]
    ages = list(range(18, 40))
    combinations = list(product(genders, nationalities, ages))*4
    tweets = []

    for combination in tqdm(combinations, desc="Generating tweets"):
        tweets.append(generate((pipe, *combination)))
    print(tweets)
    create_csv(tweets)


if __name__ == "__main__":
    main()
