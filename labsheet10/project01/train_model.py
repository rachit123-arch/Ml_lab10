import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

df = pd.read_csv("loan_approval_dataset.csv")
X, y = df.drop(columns=["Loan_Status"]), df["Loan_Status"]
num = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat = X.select_dtypes(include=["object"]).columns.tolist()
prep = ColumnTransformer([
 ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num),
 ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat)
])
models = {
 "Logistic Regression": LogisticRegression(max_iter=1500,class_weight="balanced"),
 "Decision Tree": DecisionTreeClassifier(max_depth=8,class_weight="balanced",random_state=42),
 "Random Forest": RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42),
 "Extra Trees": ExtraTreesClassifier(n_estimators=200,class_weight="balanced",random_state=42),
 "SVM": SVC(class_weight="balanced",probability=True,random_state=42)
}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
results=[]; fitted={}
for name,model in models.items():
    pipe=Pipeline([("preprocessor",prep),("model",model)])
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); fitted[name]=pipe
    results.append({"Algorithm":name,"Accuracy":accuracy_score(yte,pred),
      "Precision":precision_score(yte,pred,pos_label="Approved",zero_division=0),
      "Recall":recall_score(yte,pred,pos_label="Approved",zero_division=0),
      "F1_Score":f1_score(yte,pred,pos_label="Approved",zero_division=0)})
result=pd.DataFrame(results).sort_values("F1_Score",ascending=False)
result.to_csv("model_comparison.csv",index=False)
best=result.iloc[0]["Algorithm"]
joblib.dump(fitted[best],"loan_approval_model.joblib")
print("\nMODEL COMPARISON\n",result.to_string(index=False))
print("\nBest model:",best)
print("\nClassification report:\n",classification_report(yte,fitted[best].predict(Xte)))
