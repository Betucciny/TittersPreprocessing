import boto3
import pandas as pd
import uuid

# Initialize the DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='your-region')

# Load the CSV file
csv_file_path = 'path/to/your/csvfile.csv'
data = pd.read_csv(csv_file_path)

# Get the table
table_name = 'Tweets'
table = dynamodb.Table(table_name)

# Iterate over the CSV rows and upload to DynamoDB
for index, row in data.iterrows():
    item = {
        'id': str(uuid.uuid4()),  # Generate a unique id
        'age': int(row['age']),
        'gender': row['gender'],
        'nationality': row['nationality'],
        'tweet': row['tweet'],
        'nationality_gender': f"{row['nationality']}_{row['gender']}"  # If using composite key approach
    }
    table.put_item(Item=item)

print("Data uploaded successfully")
