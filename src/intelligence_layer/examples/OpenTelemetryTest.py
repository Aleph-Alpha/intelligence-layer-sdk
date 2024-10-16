import time
from datetime import datetime

import requests
from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.post("/print_json")
async def print_json(request: Request):
    json_data = await request.json()
    with open('test.jsonl', 'a') as file:
        file.write(str(datetime.now()) + '\t' + json.dumps(json_data) + '\n')
    return {"message": "JSON received and logged"}


