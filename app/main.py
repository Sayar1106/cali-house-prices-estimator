from fastapi import FastAPI, Form  
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

templates = Jinja2Templates(directory="./app/templates")
app = FastAPI()

pipe = joblib.load("./model/model.joblib")

@app.get("/home/", response_class=HTMLResponse)
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