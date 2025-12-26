from fastapi import FastAPI, Form
from pydantic import BaseModel
import pickle
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

model = pickle.load(open("model.pkl", "rb"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentData(BaseModel):
    study_hours: int
    attendance: int
    score: float


@app.get("/")
def homepage():
    return FileResponse("index.html")


@app.post("/predict")
def predict_result(study_hours: int = Form(...),
                   attendance: int = Form(...),
                   score: int = Form(...)
                   ):
    prediction = model.predict(
        [[study_hours, attendance, score]])

    result = "PASS" if prediction[0] == 1 else "FAIL"

    return {"prediction": result}


 