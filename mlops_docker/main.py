from fastapi import FastAPI, Request, Depends
from sqlmodel import Session
from database import engine, get_db, create_db_and_tables
from models import LoanApplication, LoanPrediction
import joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "best_loan_pipeline.pkl")
pipeline_model = joblib.load(MODEL_PATH)

# Veritabanı tablolarını oluştur
create_db_and_tables()

def make_loan_prediction(model, request: dict):
    df_input = pd.DataFrame([request])
    if "Property_Area" not in df_input.columns or df_input.Property_Area.isnull().any():
        df_input["Property_Area"] = "Urban"
    prediction = model.predict(df_input)
    return int(prediction[0])

def insert_loan_prediction(request, prediction, client_ip, db):
    new_prediction = LoanPrediction(
        gender=request["Gender"],
        married=request["Married"],
        dependents=request["Dependents"],
        education=request["Education"],
        self_employed=request["Self_Employed"],
        loan_amount_log=request["LoanAmount_log"],
        loan_amount_term=request["Loan_Amount_Term"],
        credit_history=request["Credit_History"],
        total_income_log=request["TotalIncome_log"],
        property_area=request["Property_Area"],
        prediction=prediction,
        client_ip=client_ip
    )
    with db as session:
        session.add(new_prediction)
        session.commit()
        session.refresh(new_prediction)
    return new_prediction

@app.post("/predict")
def predict_loan_status(request: LoanApplication, fastapi_req: Request, db: Session = Depends(get_db)):
    prediction_label = make_loan_prediction(pipeline_model, request.dict())
    db_insert_record = insert_loan_prediction(
        request=request.dict(),
        prediction=prediction_label,
        client_ip=fastapi_req.client.host,
        db=db
    )
    return {
        "loan_status_prediction": prediction_label,
        "meaning": "Approved" if prediction_label == 1 else "Rejected",
        "db_record": db_insert_record
    }

@app.get("/client")
def client_info(request: Request):
    return {
        "client_host": request.client.host,
        "client_port": request.client.port
    }

@app.get("/")
async def root():
    return {"data": "Welcome to MLOps Loan Prediction API"}
