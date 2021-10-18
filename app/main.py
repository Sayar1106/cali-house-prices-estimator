from fastapi import FastAPI, Form  
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import boto3
import joblib
import os

BUCKET_NAME = "california-housing-model"
FILE_NAME = "model.joblib"
LOCAL_PATH = "./model/model.joblib"
templates = Jinja2Templates(directory="./app/templates")
app = FastAPI()


def load_model():
    sess = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    client = sess.resource('s3')
    client.Bucket(BUCKET_NAME).download_file(
        FILE_NAME, LOCAL_PATH
    )
    return joblib.load(LOCAL_PATH)

pipe = load_model()
os.remove(LOCAL_PATH)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submitform")
async def handle_form(
    request: Request,
    latitude: float = Form(...), 
    longitude: float = Form(...), 
    housing_median_age: float = Form(...),
    total_rooms: float = Form(...),
    total_bedrooms: float = Form(...),
    population: float = Form(...),
    households: float = Form(...),
    median_income: float = Form(...),
    ocean_proximity: str = Form(...),
    ):

    pred_dict = {
       "latitude": latitude,
       "longitude": longitude,
       "housing_median_age": housing_median_age,
       "total_rooms": total_rooms,
       "total_bedrooms": total_bedrooms,
       "population": population,
       "households": households,
       "median_income": median_income,
       "ocean_proximity": ocean_proximity,
    }
    price = predict(pred_dict)[0]

    return templates.TemplateResponse("result.html", {"request": request, "price": price})


def predict(pred_dict):
    p_df = pd.DataFrame([pred_dict])
    p_df["rooms_per_household"] = p_df["total_rooms"]/p_df["households"]
    p_df["population_per_household"] = p_df["population"]/p_df["households"]

    return pipe.predict(p_df)