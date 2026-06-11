from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model

app = FastAPI(title="Telco Churn Prediction API", version="1.0")

model = load_model("final_telco_churn_gbc_model")

class CustomerData(BaseModel):
    Contract: str
    Tenure_Months: int
    Internet_Service: str
    Monthly_Charges: float

@app.get("/")
def home():
    return {"message": "Process is successfull"}

@app.post("/predict")
def predict_churn(customer: CustomerData):

    scaled_tenure = (customer.Tenure_Months - 32.422) / 24.547
    scaled_monthly = (customer.Monthly_Charges - 64.800) / 30.083
    
    if customer.Internet_Service == "Fiber optic":
        is_fiber = 1
        is_no = 0
    elif customer.Internet_Service == "No":
        is_fiber = 0
        is_no = 1
    else:
        is_fiber = 0
        is_no = 0

    if customer.Contract == "Month-to-month":
        contract_numeric = 0
    elif customer.Contract == "One year":
        contract_numeric = 1
    else:
        contract_numeric = 2

    input_data = pd.DataFrame([{"Contract": contract_numeric, "Tenure Months": scaled_tenure, "Monthly Charges": scaled_monthly,
                                "Internet Service_Fiber optic": is_fiber, "Internet Service_No": is_no,
                                "Latitude": 0.0, "Longitude": 0.0, "Gender": 0, "Senior Citizen": 0,
                                "Partner": 0, "Dependents": 0, "Phone Service": 0, "Multiple Lines": 0,
                                "Online Security": 0, "Online Backup": 0, "Device Protection": 0,
                                "Tech Support": 0, "Streaming TV": 0, "Streaming Movies": 0,
                                "Paperless Billing": 0, "Payment Method_Credit card (automatic)": 0,
                                "Payment Method_Electronic check": 0, "Payment Method_Mailed check": 0}])
    
    predictions = predict_model(model, data=input_data)
    prediction_label = int(predictions["prediction_label"].iloc[0])
    prediction_score = float(predictions["prediction_score"].iloc[0])
    
    status = "HIGH RISK" if prediction_label == 1 else "SAFE"
    
    return {
        "churn_risk_status": status,
        "confidence_score": prediction_score
    }
