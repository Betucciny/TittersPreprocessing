import torch
from transformers import pipeline
import pandas as pd


def setup():
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    torch_dtype=torch.bfloat16, device_map="auto")
    return pipe


def generate(pipe, gender, nationality, age):
    input = f"Write me  a daily tweet intended to make me laugh, it could be anything like you would post on sotial media. Focus on witty observations or humorous takes on everyday situations. The tweet should be concise, clever, and include a punchline at the end. Avoid using emojis, rely solely on words for humor. You don't have to explain the joke I'll get it. Thanks!"
    messages = [
        {
            "role": "system",
            "content": f"Hi there I'm a {gender} {nationality}, I'm {age} years old and I'm here to help you with your daily tweet. I'll do my best to make you laugh!",
        },
        {
            "role": "user",
            "content": input,
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True,
                   temperature=0.7, top_k=50, top_p=0.95)
    tweet = outputs[0]["generated_text"].split("<|assistant|>")[1].strip()
    return tweet


def create_csv(tweets):
    df = pd.DataFrame(tweets)
    df.to_csv("tweets.csv")


def main():
    pipe = setup()
    genders = ["male", "female", "non-binary"]
    nationalities = ["American", "British",
                     "Canadian", "Australian", "Indian", "Mexican"]
    ages = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    combinations = [(gender, nationality, age)
                    for gender in genders for nationality in nationalities for age in ages]
    finaltweets = []
    for i in range(10):
        for combination in combinations:
            finaltweets.append({"age": combination[2], "nationality": combination[1], "gender": combination[0], "tweet": generate(
                pipe, combination[0], combination[1], combination[2])})
            
        break
    create_csv(finaltweets)
    


if __name__ == "__main__":
    main()
