import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Disease Prediction Demo",page_icon="🩺")
st.title("🩺 Disease Prediction System")
st.error("Educational demonstration only. This is NOT a medical diagnostic tool. Do not use it for treatment or healthcare decisions. Consult a qualified medical professional.")
if not Path("disease_prediction_model.joblib").exists():
    st.error("Model missing. Run `python train_model.py` first."); st.stop()
model=joblib.load("disease_prediction_model.joblib")
symptoms=["cough","sneezing","runny_nose","sore_throat","mild_fever","fever","body_ache",
"headache","fatigue","chills","nausea","light_sensitivity","vomiting","dizziness","itchy_eyes",
"skin_rash","watery_eyes","diarrhea","stomach_pain","breathlessness","wheezing","chest_tightness",
"frequent_urination","increased_thirst","increased_hunger","blurred_vision","muscle_pain",
"nasal_congestion","loss_of_taste","abdominal_cramps","dry_mouth"]
with st.form("symptom_form"):
    c1,c2=st.columns(2)
    with c1:
        age=st.number_input("Age",min_value=1,max_value=110,value=25)
        gender=st.selectbox("Gender",["Female","Male","Other"])
        temp=st.number_input("Temperature (°C)",min_value=34.0,max_value=43.0,value=37.0,step=.1)
    with c2:
        bp=st.selectbox("Blood pressure category",["Normal","Elevated","High"])
    selected=st.multiselect("Select symptoms",symptoms,format_func=lambda s:s.replace("_"," ").title())
    submit=st.form_submit_button("Run educational prediction")
if submit:
    data={"Age":age,"Gender":gender,"Temperature_C":temp,"Blood_Pressure_Category":bp}
    data.update({s:int(s in selected) for s in symptoms})
    sample=pd.DataFrame([data])
    pred=model.predict(sample)[0]
    st.info("Predicted dataset class: "+str(pred))
    if hasattr(model,"predict_proba"):
        vals=model.predict_proba(sample)[0]
        scores=pd.DataFrame({"Dataset class":model.classes_,"Model score":vals}).sort_values("Model score",ascending=False).head(5)
        st.write("Top model scores (not clinical probabilities):")
        st.dataframe(scores,use_container_width=True)
