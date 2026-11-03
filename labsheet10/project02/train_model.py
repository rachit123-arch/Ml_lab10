import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

df=pd.read_csv("disease_symptoms_dataset.csv")
X,y=df.drop(columns=["Disease"]),df["Disease"]
num=X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat=X.select_dtypes(include=["object"]).columns.tolist()
numeric=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
prep=ColumnTransformer([("num",numeric,num),("cat",categorical,cat)])
models={
 "Decision Tree":DecisionTreeClassifier(max_depth=12,random_state=42),
 "Random Forest":RandomForestClassifier(n_estimators=200,random_state=42),
 "Extra Trees":ExtraTreesClassifier(n_estimators=200,random_state=42),
 "K-Nearest Neighbors":KNeighborsClassifier(n_neighbors=7),
 "Gaussian Naive Bayes":GaussianNB(),
 "SVM":SVC(probability=True,random_state=42)
}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
results=[]; fitted={}
for name,model in models.items():
    pipe=Pipeline([("preprocessor",prep),("model",model)])
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); fitted[name]=pipe
    results.append({"Algorithm":name,"Accuracy":accuracy_score(yte,pred),
      "Precision_Macro":precision_score(yte,pred,average="macro",zero_division=0),
      "Recall_Macro":recall_score(yte,pred,average="macro",zero_division=0),
      "F1_Macro":f1_score(yte,pred,average="macro",zero_division=0)})
result=pd.DataFrame(results).sort_values("F1_Macro",ascending=False)
result.to_csv("model_comparison.csv",index=False)
best=result.iloc[0]["Algorithm"]
joblib.dump(fitted[best],"disease_prediction_model.joblib")
print("\nMODEL COMPARISON\n",result.to_string(index=False))
print("\nBest model:",best)
print("\nClassification report:\n",classification_report(yte,fitted[best].predict(Xte)))
