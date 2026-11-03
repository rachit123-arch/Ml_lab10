import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Loan Approval Prediction",page_icon="🏦")
st.title("🏦 Loan Approval Prediction System")
st.warning("Educational demonstration using synthetic data. Not a real lending decision.")
if not Path("loan_approval_model.joblib").exists():
    st.error("Model missing. Run `python train_model.py` first."); st.stop()
model=joblib.load("loan_approval_model.joblib")
with st.form("loan_form"):
    c1,c2=st.columns(2)
    with c1:
        gender=st.selectbox("Gender",["Male","Female"])
        married=st.selectbox("Married",["Yes","No"])
        dependents=st.selectbox("Dependents",["0","1","2","3+"])
        education=st.selectbox("Education",["Graduate","Not Graduate"])
        self_emp=st.selectbox("Self-employed",["Yes","No"])
        applicant_income=st.number_input("Applicant monthly income",min_value=0,value=5000,step=500)
    with c2:
        co_income=st.number_input("Co-applicant monthly income",min_value=0,value=0,step=500)
        loan_amount=st.number_input("Loan amount (thousands)",min_value=1,value=120,step=5)
        loan_term=st.selectbox("Loan term (months)",[120,180,240,300,360],index=4)
        credit=st.selectbox("Credit history",[1,0],format_func=lambda x:"Good (1)" if x==1 else "Poor/No history (0)")
        area=st.selectbox("Property area",["Urban","Semiurban","Rural"])
    submit=st.form_submit_button("Predict Loan Status")
if submit:
    row=pd.DataFrame([{"Gender":gender,"Married":married,"Dependents":dependents,
      "Education":education,"Self_Employed":self_emp,"Applicant_Income":applicant_income,
      "Coapplicant_Income":co_income,"Loan_Amount":loan_amount,"Loan_Term":loan_term,
      "Credit_History":credit,"Property_Area":area}])
    prediction=model.predict(row)[0]
    scores=model.predict_proba(row)[0]
    confidence=float(scores[list(model.classes_).index(prediction)])*100
    st.subheader("Prediction: "+prediction)
    st.metric("Model score for predicted class",f"{confidence:.1f}%")
    st.caption("Model score is not a guarantee or a bank decision.")
