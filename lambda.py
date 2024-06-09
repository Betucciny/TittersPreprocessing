import boto3
import json

print('Loading function')
dynamo = boto3.client('dynamodb')


def respond(err, res=None):
    return {
        'statusCode': '400' if err else '200',
        'body': err.message if err else json.dumps(res),
        'headers': {
            'Content-Type': 'application/json',
        },
    }


def lambda_handler(event, context):
    '''Demonstrates a simple HTTP endpoint using API Gateway. You have full
    access to the request and response payload, including headers and
    status code.

    To scan a DynamoDB table, make a GET request with the TableName as a
    query string parameter. To put, update, or delete an item, make a POST,
    PUT, or DELETE request respectively, passing in the payload to the
    DynamoDB API as a JSON body.
    '''
    # print("Received event: " + json.dumps(event, indent=2))

    payload = event['queryStringParameters']
    age = payload['age']
    nationality = payload['nationality']
    gender = payload['gender']
    
    # Query the DynamoDB table using the age and nationality_gender composite key
    data = {
        'TableName': 'Tweets',
        'IndexName': 'AgeNationalityGenderIndex',  # Assuming this is the GSI for age and nationality_gender
        'KeyConditionExpression': 'age = :age and nationality_gender = :nationality_gender',
        'ExpressionAttributeValues': {
            ':age': {'N': age},
            ':nationality_gender': {'S': f"{nationality}_{gender}"}
        },
        'Limit': 1,  # Limit the number of results to 10
    }
    result = dynamo.query(**data)
    id = result['Items'][0]['id']['S']
    trydelete = dynamo.delete_item(
        TableName='Tweets',
        Key={
            'id': {'S': id},
        }
    )
    tweet = result['Items'][0]['tweet']['S']
    return respond(None, tweet)


def main():
    response = lambda_handler(
        {'queryStringParameters':
            {'age': '20', 'nationality': 'American', 'gender': 'male'}
         }, None)
    print(response)


if __name__ == '__main__':
    main()
