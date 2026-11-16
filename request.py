import requests

url = "http://localhost:9696/predict"

passenger = {
    "PassengerId": 892,
    "Pclass": 3,
    "Name": "Kelly, Mr. James",
    "Sex": "male",
    "Age": 34.5,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": "330911",
    "Fare": 7.8292,
    "Cabin": None,
    "Embarked": "Q"
}

response = requests.post(url, json=passenger)
predictions = response.json()

print(predictions)
if predictions['survived']:
    print('passenger is likely to survive')
else:
    print('passenger is not likely to survive')