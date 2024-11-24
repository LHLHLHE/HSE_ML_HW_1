import io
import joblib
import re
from fastapi import FastAPI,  UploadFile, File
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np


app = FastAPI()

model = joblib.load('ridge_pipline.pkl')


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

def extract_nm(value):
    if pd.isnull(value):
        return np.nan
    value = value.lower()

    nm_match = re.search(r'(\d+\.?\d*)nm', value)
    if nm_match:
        return float(nm_match.group(1))
    
    # Если Nm нет, ищем kgm и переводим в Nm
    kgm_match = re.search(r'(\d+\.?\d*)kgm', value)
    if kgm_match:
        return float(kgm_match.group(1)) * 9.8
    return np.nan

def extract_rpm(value):
    if pd.isnull(value):
        return np.nan
    value = value.lower()
    
    # Если есть диапазон rpm, берем среднее
    range_match = re.search(r'(\d+)-(\d+)rpm', value)
    if range_match:
        start, end = map(int, range_match.groups())
        return (start + end) / 2

    rpm_match = re.search(r'(\d+)rpm', value)
    if rpm_match:
        return int(rpm_match.group(1))
    return np.nan


def preprocess_columns(df):
    df.loc[df['mileage'].str.contains('km/kg', na=False), 'mileage'] = np.nan

    df['mileage'] = df['mileage'].str.removesuffix('kmpl')
    df['mileage_kmpl'] = pd.to_numeric(df['mileage'], errors='coerce', downcast='float')
    df.drop(columns=['mileage'], inplace=True)

    df['engine'] = df['engine'].str.removesuffix(' CC')
    df['engine_cc'] = pd.to_numeric(df['engine'], errors='coerce', downcast='float')
    df.drop(columns=['engine'], inplace=True)

    df['max_power'] = df['max_power'].str.removesuffix(' bhp')
    df['max_power_bhp'] = pd.to_numeric(df['max_power'], errors='coerce', downcast='float')
    df.drop(columns=['max_power'], inplace=True)

    df['torque_nm'] = df['torque'].apply(extract_nm)
    df['rpm'] = df['torque'].apply(extract_rpm)
    df.drop(columns=['torque'], inplace=True)

    df['name'] = df['name'].apply(lambda x: x.split()[0])


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.model_dump()]).drop(columns=['selling_price'])
    preprocess_columns(data)
    return model.predict(data)[0]


@app.post("/predict_items")
def predict_items(file: UploadFile = File()) -> StreamingResponse:
    data = pd.read_csv(file.file)
    
    data_for_predict = data.drop(columns=['selling_price']).copy()
    preprocess_columns(data_for_predict)
    preds = model.predict(data_for_predict)

    data['predicted_price'] = preds

    output = io.StringIO()
    data.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output, media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predicted_{file.filename}"}
    )