from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel
from typing import Optional
app = FastAPI()

# information que je te donne en plus leo : 
# le model ne doit pas etre réentrainer a chaque demarage de l'api mais seulement une fois, 
# au meme niveau que les données je penses que tu pourrais mettre un fichier de configuration du model avec la sauvegrade

# inisialisation of class model
# (leo)

# this function represents a sequence of predicted winning numbers of the model
@app.get('/api/perdit')
async def get_predict():
    return "1, 2, 3, 4, 5 | 6, 7"

# this function gives the percentage of success and loss of the set of numbers proposed in parameter 
@app.post('/api/perdit')
async def post_predict(n1, n2, n3, n4, n5, e1, e2):
    return f"fonction predit que {n1, n2, n3, n4, n5, e1, e2} a 50% de chance de gain et 50% de chance de perte"

# this function this function gives the characteristics of the predictive model
@app.get('/api/model')
async def get_model():
    return "caracteristique qu model"

# this function adds data to the model
@app.put('/api/model/{data}')
async def put_model(data):
    return f"ajoute {data} au model"

# this function entertains the model with the new data
@app.post('/api/model/retrain')
async def retrain():
    return "nouvelle entrainement des données"