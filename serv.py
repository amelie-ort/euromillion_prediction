from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel
from typing import Optional
app = FastAPI()


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"

@app.get('/')
async def root():
    return {"message":"hello world"}

@app.get("/items/{item_id}")
async def read_item(item_id : int):
    return {"item_id": item_id}

@app.get("/models/{model_name}")
async def get_model(model_name : ModelName):
    if model_name == ModelName.alexnet: 
       return {"model_name": model_name} 
    if model_name.value == "resnet":
       return {"model_name": model_name} 
    return {"model_name": model_name}

class Item(BaseModel): 
    name: int
    description : Optional[str] = None
    price: float

@app.post("/items/")
async def create_item(item : Item):
    return item

