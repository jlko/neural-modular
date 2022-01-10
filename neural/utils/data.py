import json


def to_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file)


def from_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data
