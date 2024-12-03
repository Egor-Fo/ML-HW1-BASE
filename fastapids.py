from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

with open('car_model_elastic.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['mileage', 'engine', 'max_power']
    for col in cols:
        df[col] = df[col].astype(str).str.extract('(\d+)').astype(float)
    df = df.drop(columns=["torque"], errors='ignore')
    median = df[df.columns[df.isnull().any()].tolist()].median()
    df = df.fillna(median)
    cols_int = ['engine', 'seats']
    for col in cols_int:
        df[col] = df[col].astype(int)

    df = df[['year','km_driven','mileage','engine','max_power','seats']]
    return df



@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.dict()])
    data = preprocess_data(data)
    X_scaled = scaler.transform(data)
    prediction = model.predict(X_scaled)
    return prediction[0]


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, sep=',',on_bad_lines='warn')
    df = preprocess_data(df)
    X_scaled = scaler.transform(df)
    predictions = model.predict(X_scaled)
    df['predicted_price'] = predictions
    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)

    return {"file": output_file}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapids:app", host="127.0.0.1", port=8000, reload=True)
