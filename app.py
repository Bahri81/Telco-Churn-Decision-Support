import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import io

st.set_page_config(page_title="Telco Churn Decision Support", layout="wide")

st.title("Telco Churn Prediction System")

st.markdown("""
*I will use Streamlit application delivers a comprehensive decision-support platform to present the selected model through dual deployment features: an Interactive Single Profile Simulator for real-time risk analysis, and a High-Volume Bulk Prediction Engine for Excel/CSV portfolios*
""")

with st.expander("View Technical Preprocessing & Pipeline Alignment Notes"):
    st.markdown("""
    To ensure that raw data can be properly utilized by the trained machine learning pipeline., the following alignment strategies were applied:
    * **Reducing Column Selection:** 4 inputs are selected to be choosen to simplify the process, selected variables are the most important features in prediction according to SHAP.
    * **Feature Alignment:** Formatted the sparse live data inputs into the exact 23-column schema needed by the PyCaret to eliminate index alignment conflicts.
    * **Mathematical Scaling:** Transformed live numerical attributes into Z-scores using the precise mean ($\mu$) and standard deviation ($\sigma$) derived from the baseline training results.
    * **Categorical Synchronization:** Mapped text-based selections (e.g., Contract types, Internet Service infrastructure) into their strict numerical and dummy-variable equivalents prior to model injection.
    """)

st.markdown("---")

tab1, tab2 = st.tabs([" Single Customer Simulator", "Mass Prediction (Excel/CSV)"])

with tab1:
    st.subheader("Interactive Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
    with col2:
        internet = st.selectbox("Internet Infrastructure", ["Fiber optic", "DSL", "No"])
        monthly = st.number_input("Monthly Charges ($)", 10.0, 150.0, 70.0)

    if st.button("Calculate Individual Risk"):
        model = load_model("final_telco_churn_gbc_model")
        
        s_tenure = (tenure - 32.422) / 24.547
        s_monthly = (monthly - 64.800) / 30.083
        c_num = 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)
        is_fiber = 1 if internet == "Fiber optic" else 0
        is_no = 1 if internet == "No" else 0

        input_data = pd.DataFrame([{"Contract": c_num, "Tenure Months": s_tenure, "Monthly Charges": s_monthly,
                                    "Internet Service_Fiber optic": is_fiber, "Internet Service_No": is_no,
                                    "Latitude": 0.0, "Longitude": 0.0, "Gender": 0, "Senior Citizen": 0,
                                    "Partner": 0, "Dependents": 0, "Phone Service": 0, "Multiple Lines": 0,
                                    "Online Security": 0, "Online Backup": 0, "Device Protection": 0,
                                    "Tech Support": 0, "Streaming TV": 0, "Streaming Movies": 0,
                                    "Paperless Billing": 0, "Payment Method_Credit card (automatic)": 0,
                                    "Payment Method_Electronic check": 0, "Payment Method_Mailed check": 0}])
        
        preds = predict_model(model, data=input_data)
        label = preds["prediction_label"].iloc[0]
        score = preds["prediction_score"].iloc[0]
        
        if label == 1:
            st.error(f" **High Churn Risk:** %{score * 100:.2f}")
        else:
            st.success(f" **Safe Customer:** %{score * 100:.2f}")

with tab2:
    st.subheader("Batch Processing Center")
    st.info("Please upload your raw customer list.")
    
    st.warning("""
    **A Quick Note on Bulk Demonstration:** I have built this batch as a *proof-of-concept prototype*. 
    For simplicity and avoiding manually mapping all 23 categorical variables from the original text, 
    the unselected features are automatically neutralized to baseline values as 0. 
    Because these omitted features include vital loyalty anchors (like tech support or online security packages), the model 
    will naturally perceive these raw profiles as higher risk, leading to frequent "High Churn" classifications. 
    *Please do not use this for real world deployments without integrating the full, end-to-end feature pipeline.*
    """)
    
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df_bulk = pd.read_csv(uploaded_file)
        else:
            df_bulk = pd.read_excel(uploaded_file)
            
        st.write(f" Loaded: **{len(df_bulk)}** rows detected in the source file.")
        
        if st.button(" Run Mass Prediction"):
            with st.spinner("Executing mass dataset pipeline mapping..."):
                model = load_model("final_telco_churn_gbc_model")
                
                model_cols = ["Contract", "Tenure Months", "Monthly Charges", "Internet Service_Fiber optic", "Internet Service_No", 
                              "Latitude", "Longitude", "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", 
                              "Multiple Lines", "Online Security", "Online Backup", "Device Protection", "Tech Support", 
                              "Streaming TV", "Streaming Movies", "Paperless Billing", "Payment Method_Credit card (automatic)", 
                              "Payment Method_Electronic check", "Payment Method_Mailed check"]
                
                processed_df = pd.DataFrame(0.0, index=np.arange(len(df_bulk)), columns=model_cols)
                
                if "Tenure Months" in df_bulk.columns:
                    processed_df["Tenure Months"] = (df_bulk["Tenure Months"] - 32.422) / 24.547
                if "Monthly Charges" in df_bulk.columns:
                    processed_df["Monthly Charges"] = (df_bulk["Monthly Charges"] - 64.800) / 30.083
                
                if "Contract" in df_bulk.columns:
                    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
                    processed_df["Contract"] = df_bulk["Contract"].map(contract_map).fillna(0)
                
                if "Internet Service" in df_bulk.columns:
                    processed_df["Internet Service_Fiber optic"] = df_bulk["Internet Service"].apply(lambda x: 1 if x == "Fiber optic" else 0)
                    processed_df["Internet Service_No"] = df_bulk["Internet Service"].apply(lambda x: 1 if x == "No" else 0)
                
                final_preds = predict_model(model, data=processed_df)
                
                df_bulk["Churn_Risk_Status"] = final_preds["prediction_label"].map({1: "HIGH RISK", 0: "SAFE"})
                df_bulk["Confidence_Score"] = final_preds["prediction_score"]
                
                st.markdown("### Process Complete!")
                st.dataframe(df_bulk[["Contract", "Tenure Months", "Monthly Charges", "Churn_Risk_Status", "Confidence_Score"]].head(15))
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df_bulk.to_excel(writer, index=False, sheet_name="Predictions")
                processed_data = output.getvalue()
                
                st.download_button(
                    label=" Download Full Prediction Report (Excel)",
                    data=processed_data,
                    file_name="Telco_Churn_Mass_Predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
