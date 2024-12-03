from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel
from pickle import load
import re
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import logging


app = FastAPI()
model_path = 'car_price_model.pkl'
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')


class Item(BaseModel):
    name: str
    year: int
    # selling_price: int
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


# class Items(BaseModel):
#     objects: List[Item]


class ConversionToFormat:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def convert_torque(self):
        def process_torque(row):
            row = row.strip().lower()

            match_nm = re.search(r'(\d+\.?\d*)\s*nm@?\s*(\d{1,5}(?:[-~]\d{1,5})?)\s*rpm', row)
            if match_nm:
                torque = float(match_nm.group(1))
                rpm_range = match_nm.group(2)
                if '-' in rpm_range:
                    rpm = int(rpm_range.split('-')[0].replace(',', ''))
                else:
                    rpm = int(rpm_range.split('~')[0].replace(',', ''))
                return torque, rpm

            match_kgm = re.search(r'(\d+\.?\d*)\s*kgm(?:\s*at\s*)?(\d{1,5}(?:[-~]\d{1,5})?)\s*rpm', row)
            if match_kgm:
                torque_kgm = float(match_kgm.group(1))
                torque = torque_kgm * 9.8
                rpm_range = match_kgm.group(2)
                if '-' in rpm_range:
                    rpm = int(rpm_range.split('-')[0].replace(',', ''))
                else:
                    rpm = int(rpm_range.split('~')[0].replace(',', ''))
                return torque, rpm

            match_kgm_at = re.search(r'(\d+\.?\d*)\s*kgm@\s*(\d{1,5}(?:[-~]\d{1,5})?)\s*rpm', row)
            if match_kgm_at:
                torque_kgm = float(match_kgm_at.group(1))
                torque = torque_kgm * 9.80665
                rpm_range = match_kgm_at.group(2)
                rpm = int(rpm_range.split('-')[0].replace(',', ''))
                return torque, rpm

            match_generic = re.search(r'(\d+\.?\d*)\s*@\s*(\d{1,5}(?:[-~]\d{1,5})?)\s*\(kgm@\s*rpm\)', row)
            if match_generic:
                torque_generic = float(match_generic.group(1))
                torque = torque_generic * 9.8
                rpm_range = match_generic.group(2)
                rpm = int(rpm_range.split('-')[0].replace(',', ''))
                return torque, rpm
            return None, None

        self.df[['torque', 'max_torque_rpm']] = self.df['torque'].apply(lambda x: pd.Series(process_torque(x)))

    def convert_mileage_engine_max_power(self):
        for col in ['mileage', 'engine', 'max_power']:
            self.df[col] = np.where(
                self.df[col].notna(),
                self.df[col].str.replace(r'[^0-9\.]', '', regex=True).replace('', 0).astype('float'),
                self.df[col]
            )
            self.df[col] = self.df[col].astype('float')

    def convert_name(self):
        self.df['name'] = self.df['name'].str.split(' ').str[0]

    def creat_average_km_per_year(self):
        self.df['average_km_per_year'] = self.df['km_driven'] / (2021 - self.df['year'])

    def creat_hp_per_liter(self):
        self.df['hp_per_liter'] = self.df['max_power'] / (self.df['engine'] / 1000)

    def transform(self):
        self.convert_torque()
        self.convert_mileage_engine_max_power()
        self.convert_name()
        self.creat_average_km_per_year()
        self.creat_hp_per_liter()
        return self.df


with open(model_path, 'rb') as model_file:
    model = load(model_file)


async def prediction_model(data: pd.DataFrame):
    converter = ConversionToFormat(data)
    df_transformed = converter.transform()
    prediction = model.predict(df_transformed)
    return prediction


async def pydantic_model_to_df(model_instance) -> pd.DataFrame:
    return pd.DataFrame([jsonable_encoder(model_instance)])


@app.get("/")
async def root():
    return {"message": "Hello!"}


@app.post("/predict_item")
async def predict_item(item: Item) -> dict:
    df = await pydantic_model_to_df(item)
    prediction = await prediction_model(df)
    return {"price": prediction.tolist()[0][0]}


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    content = await file.read()
    decoded_content = content.decode('utf-8')

    df = pd.read_csv(StringIO(decoded_content))

    rows = df.to_dict(orient='records')

    logging.info(f"{rows}")

    df = pd.DataFrame([jsonable_encoder(Item.model_validate(row)) for row in rows])
    prediction = await prediction_model(df)
    df['selling_price'] = prediction

    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})


# print(jsonable_encoder(Item.model_validate({'name': 'Maruti Swift Dzire VDI', 'year': 2014, 'km_driven': 145500, 'fuel': 'Diesel', 'seller_type': 'Individual', 'transmission': 'Manual', 'owner': 'First Owner', 'mileage': '23.4 kmpl', 'engine': '1248 CC', 'max_power': '74 bhp', 'torque': '190Nm@ 2000rpm', 'seats': 5.0})))
