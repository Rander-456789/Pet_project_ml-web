import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

class ClientData(BaseModel):
    age: int
    income: float
    education: bool
    work: bool
    car: bool

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

model = joblib.load('model.pkl')

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.post('/score')
def score(data: ClientData):
    features = [data.age, data.income, data.education, data.work, data.car]
    approved = not model.predict([features])[0].item()
    return {'approved': approved}

