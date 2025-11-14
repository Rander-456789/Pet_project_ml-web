import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware



class ClientData(BaseModel):
    age: int
    income: float
    education: bool
    work: bool
    car: bool

app = FastAPI()
model = joblib.load('model.pkl')

@app.post('/score')
def score(data: ClientData):
    features = [data.age, data.income, data.education, data.work, data.car]
    approved = not model.predict([features])[0].item()
    return {'approved': approved}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)