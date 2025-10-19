from typing import Tuple, List
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import shap

def train_promotion_model(df: pd.DataFrame) -> Tuple[Pipeline, float]:
    # Create a synthetic label: 'promote' if high perf & engagement & learning/tenure
    y = ( (df["performance"]>0.7) & (df["engagement"]>0.65) & (df["learning_speed"]>0.55) & (df["tenure_months"]>12) ).astype(int)
    X = df[["gender","dept","age","tenure_months","performance","engagement","learning_speed"]]
    cat = ["gender","dept"]
    num = ["age","tenure_months","performance","engagement","learning_speed"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    return model, acc

def shap_explain(model: Pipeline, df: pd.DataFrame, rows: List[int]):
    # Use KernelExplainer for pipeline
    X = df[["gender","dept","age","tenure_months","performance","engagement","learning_speed"]]
    f = lambda data: model.predict_proba(pd.DataFrame(data, columns=X.columns))[:,1]
    background = X.sample(min(60, len(X)), random_state=1)
    explainer = shap.KernelExplainer(f, background, link="logit")
    shap_values = explainer.shap_values(X.iloc[rows], nsamples=100)
    base = explainer.expected_value
    return shap_values, base
