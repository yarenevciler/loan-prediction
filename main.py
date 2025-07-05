from fastapi import FastAPI, Request
from models import LoanApplication
import joblib
import pandas as pd
import os

app = FastAPI()


MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "best_loan_pipeline.pkl")
pipeline_model = joblib.load(MODEL_PATH)



def make_loan_prediction(model, request: dict):
    """
    Tahmin fonksiyonu: request dict'ini alır, DataFrame'e çevirir ve tahmin döner.
    """
    df_input = pd.DataFrame([request])

    # Property_Area boş ise default ayarla
    if "Property_Area" not in df_input.columns or df_input.Property_Area.isnull().any():
        df_input["Property_Area"] = "Urban"

    prediction = model.predict(df_input)
    return int(prediction[0])


@app.post("/predict")
def predict_loan_status(request: LoanApplication):
    """
    Tahmin endpointi: request alır, tahmin fonksiyonuna gönderir ve sonucu döner.
    """
    prediction_label = make_loan_prediction(pipeline_model, request.dict())
    return {
        "loan_status_prediction": prediction_label,
        "meaning": "Approved" if prediction_label == 1 else "Rejected"
    }


@app.get("/client")
def client_info(request: Request):
    """
    Client bilgisi endpointi: istek yapanın IP ve port bilgisini döner.
    """
    return {
        "client_host": request.client.host,
        "client_port": request.client.port
    }
