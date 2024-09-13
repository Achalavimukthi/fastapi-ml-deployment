# app.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load the California Housing model
model = joblib.load("california_model.pkl")

# Define the input schema
class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Define a prediction endpoint
@app.post("/predict/")
def predict(data: HousingData):
    input_data = [[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                   data.Population, data.AveOccup, data.Latitude, data.Longitude]]
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}
