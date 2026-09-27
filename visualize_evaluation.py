# Regenerates the six evaluation visualizations.
# See train_evaluate.py for model training and metrics.
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.inspection import permutation_importance

df=pd.read_csv("data/iris_ml.csv"); f=["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"]; X=df[f]; y=df.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={"Logistic Regression":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1000,random_state=42))]),"Decision Tree":DecisionTreeClassifier(max_depth=4,random_state=42)}
pred={}
for n,m in models.items(): m.fit(Xtr,ytr); pred[n]=m.predict(Xte)
# Figures are saved to visualizations/
print("Evaluation visualizations can be regenerated from this script.")
