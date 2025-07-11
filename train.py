import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
import warnings
import socket

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5001")

os.makedirs("saved_models", exist_ok=True)

# 🔢 Veri işlemleri
df = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')

if 'Loan_ID' in df.columns:
    df.drop(['Loan_ID'], axis=1, inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
if 'Married' in df.columns:
    df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
if 'Education' in df.columns:
    df['Education'] = df['Education'].map({'Not Graduate': 0, 'Graduate': 1})
if 'Self_Employed' in df.columns:
    df['Self_Employed'] = df['Self_Employed'].map({'No': 0, 'Yes': 1})
if 'Dependents' in df.columns:
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
if 'Loan_Status' in df.columns:
    df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
df['TotalIncome_log'] = np.log1p(df['TotalIncome'])

df.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1, inplace=True)

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

numeric_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'LoanAmount_log', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome_log']
categorical_features = ['Property_Area'] if 'Property_Area' in df.columns else []

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# 🔁 Model tanımları
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

models = {
    "logistic_regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "xgboost": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss', use_label_encoder=False)
}

mlflow.set_experiment("loan-approval-classification-pipeline")

best_model_name = None
best_pipeline = None
best_score = 0.0

for name, clf in models.items():
    with mlflow.start_run(run_name=name) as run:
        print(f"\n====== Model: {name} ======")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"F1 Score: {f1:.4f}")

        mlflow.log_param("model_name", name)
        mlflow.log_metric("f1_score", f1)

        # Hiperparametreler
        if isinstance(clf, RandomForestClassifier):
            mlflow.log_param("n_estimators", clf.n_estimators)
            mlflow.log_param("max_depth", clf.max_depth)
        elif isinstance(clf, LogisticRegression):
            mlflow.log_param("C", clf.C)
        elif isinstance(clf, XGBClassifier):
            mlflow.log_param("learning_rate", clf.learning_rate)

        # ✅ Model loglama + kayıt (Model Registry)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=f"model-{name}",
            registered_model_name=f"{name}_model"
        )


        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_pipeline = pipeline

        joblib.dump(pipeline, f"saved_models/loan_{name}_pipeline.pkl")

# En iyi modeli kaydet
joblib.dump(best_pipeline, f"saved_models/best_loan_pipeline.pkl")
print(f"\nBest Model: {best_model_name} (F1: {best_score:.4f})")
