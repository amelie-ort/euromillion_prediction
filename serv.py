from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel
from typing import Optional
app = FastAPI()
from random import randint
from model.model import IAModel

# inisialisation of class model
m = IAModel()

@app.get('/api/perdit')
async def get_predict():
    """this function represents a sequence of predicted winning numbers of the model"""
    def genere_numbers():
        n1 = randint(1,50)
        n2 = randint(1,50) 
        n3 = randint(1,50)
        n4 = randint(1,50)
        n5 = randint(1,50)
        e1 = randint(1,12) 
        e2 = randint(1,12)
        return [n1, n2, n3, n4, n5, e1, e2]
    serie = genere_numbers()
    while m.predict_proba(serie)[1] < 0.5 :
        serie = genere_numbers()
    return serie

@app.post('/api/perdit')
async def post_predict(n1, n2, n3, n4, n5, e1, e2):
    """this function gives the percentage of success and loss of the set of numbers proposed in parameter 
    param n1, n2, n3, n4, n5, e1, e2 -> numbers 
    return proba_win, proba_loss"""
    return m.predict_proba([n1, n2, n3, n4, n5, e1, e2])[1], m.predict_proba([n1, n2, n3, n4, n5, e1, e2])[0]

@app.get('/api/model')
async def get_model():
    """this function this function gives the characteristics of the predictive model (score, name, params)"""
    return m.get_params()

@app.put('/api/model/{data}')
async def put_model(data):
    """this function adds data to the model, param : format : line of csv"""
    return f"ajoute {m.data_manager.add(data)} au model"

# this function entertains the model with the new data
@app.post('/api/model/retrain')
async def retrain():
    m.train()
    return "nouvelle entrainement des données"
