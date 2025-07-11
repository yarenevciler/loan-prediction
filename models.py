from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class LoanApplication(SQLModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    LoanAmount_log: float
    Loan_Amount_Term: float
    Credit_History: float
    TotalIncome_log: float
    Property_Area: Optional[str] = "Urban"  

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": 0,
                "Married": 1,
                "Dependents": 0,
                "Education": 1,
                "Self_Employed": 0,
                "LoanAmount_log": 4.5,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0,
                "TotalIncome_log": 10.5,
                "Property_Area": "Urban"
            }
        }
class LoanPrediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    gender: int
    married: int
    dependents: int
    education: int
    self_employed: int
    loan_amount_log: float
    loan_amount_term: float
    credit_history: float
    total_income_log: float
    property_area: str
    prediction: int
    prediction_time: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    client_ip: str
