import json

def json_load() -> tuple:
    with open('shape.json') as file_json:
        data = json.load(file_json)
    return data

def json_save(shape: int, type_: str) -> None:
    with open('shape.json') as file_json:
        data = json.load(file_json)
    data[type_] = shape
    with open('shape.json', 'w') as file_json:
        json.dump(data, file_json, indent=4)
