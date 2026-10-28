
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="House Price Prediction API")

model_path = Path(__file__).parent / "house_price.joblib"
saved = joblib.load(model_path)
model = saved["model"]

feature_names = list(
    model.named_steps["preprocessor"].feature_names_in_
)


class HouseData(BaseModel):
    features: dict[str, float | str | None]


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: HouseData):
    unknown = set(data.features) - set(feature_names)

    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown features: {sorted(unknown)}"
        )

    row = {name: None for name in feature_names}
    row.update(data.features)

    input_data = pd.DataFrame([row], columns=feature_names)
    prediction = model.predict(input_data)[0]

    return {
        "predicted_sale_price": round(float(prediction), 2)
    }
