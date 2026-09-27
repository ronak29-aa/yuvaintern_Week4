import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

df=pd.read_csv("data/iris_ml.csv")
features=["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"]
X=df[features]; y=df["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)

models={
"Logistic Regression":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1000,random_state=42))]),
"Decision Tree":DecisionTreeClassifier(max_depth=4,random_state=42)
}
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    print("\n",name)
    print("Accuracy:",accuracy_score(y_test,pred))
    print("Precision:",precision_score(y_test,pred,average="weighted"))
    print("Recall:",recall_score(y_test,pred,average="weighted"))
    print("F1:",f1_score(y_test,pred,average="weighted"))
    print(classification_report(y_test,pred,target_names=["setosa","versicolor","virginica"]))
    print("CV accuracy:",cross_val_score(model,X,y,cv=cv,scoring="accuracy"))
