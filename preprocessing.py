import pandas as pd
import json


def load_data():
    data = pd.read_csv('data/QA1.csv')
    data = data[['Question', 'Answer', 'dirt']].dropna()
    json_data = []
    for index, row in data.iterrows():
        prompt = f"This joke has an offensive level beetween 0 and 1 of: {row['dirt']}"
        response = row['Question'] + ' ' + row['Answer']
        json_data.append({'prompt': prompt, 'response': response})
        if index == 40000:
            break
    return json_data


def main():
    data = load_data()
    print(data)
    with open('data/data.json', 'w') as f:
        json.dump(data, f)

    
if __name__ == "__main__":
    main()
