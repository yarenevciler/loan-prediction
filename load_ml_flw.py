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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
import warnings

warnings.filterwarnings("ignore")  # sklearn ve xgboost uyarıları için

os.makedirs("saved_models", exist_ok=True)

# 1) Veriyi oku
df = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
print("Veri seti şekli:", df.shape)

# 2) ID kolonunu sil
if 'Loan_ID' in df.columns:
    df.drop(['Loan_ID'], axis=1, inplace=True)

# 3) Eksik değer doldurma
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_cols:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)

for col in numerical_cols:
    median = df[col].median()
    df[col] = df[col].fillna(median)

# 4) Kategorik değişkenleri binary kodla
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

# Property_Area için One-Hot Encoding yapılacak
# Diğer kategorikler binary kodlandığı için numeric_features'a eklenebilir

# Yeni özellik: Toplam gelir ve log dönüşümü
if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

if 'LoanAmount' in df.columns:
    df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
if 'TotalIncome' in df.columns:
    df['TotalIncome_log'] = np.log1p(df['TotalIncome'])

# Kullanılmayan sütunları sil
drop_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome']
df.drop([col for col in drop_cols if col in df.columns], axis=1, inplace=True)

# Eksik değer kontrolü
print("\nKalan eksik değer sayısı:", df.isnull().sum().sum())

# Özellik ve hedef ayır
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train/test böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# Numeric ve categorical feature listesi
numeric_features = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'LoanAmount_log', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome_log'
]

categorical_features = ['Property_Area'] if 'Property_Area' in df.columns else []

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Modeller
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"\nClass distribution -> Negative: {neg}, Positive: {pos}, scale_pos_weight: {scale_pos_weight:.2f}")

models = {
    "logistic_regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "xgboost": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss', use_label_encoder=False)
}

# MLflow deneyini başlat
mlflow.set_experiment("loan-approval-classification-pipeline")

best_model_name = None
best_pipeline = None
best_score = 0.0

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\n====== Model: {name} ======")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print(classification_report(y_test, y_pred))
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"F1 Score: {f1:.4f}")

        mlflow.log_param("model_name", name)
        mlflow.log_metric("f1_score", f1)

        # 🔥 Hiperparametreleri MLflow'a logla
        if isinstance(clf, RandomForestClassifier):
            mlflow.log_param("n_estimators", clf.n_estimators)
            mlflow.log_param("max_depth", clf.max_depth)
            mlflow.log_param("max_features", clf.max_features)
        elif isinstance(clf, LogisticRegression):
            mlflow.log_param("C", clf.C)
            mlflow.log_param("max_iter", clf.max_iter)
            mlflow.log_param("penalty", clf.penalty)
        elif isinstance(clf, XGBClassifier):
            mlflow.log_param("max_depth", clf.max_depth)
            mlflow.log_param("learning_rate", clf.learning_rate)
            mlflow.log_param("scale_pos_weight", clf.scale_pos_weight)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=f"model-{name}",
            registered_model_name=None
        )

        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_pipeline = pipeline

        joblib.dump(pipeline, f"saved_models/loan_{name}_pipeline.pkl")
        print(f"Pipeline saved as 'saved_models/loan_{name}_pipeline.pkl'")

# En iyi pipeline'ı kaydet
joblib.dump(best_pipeline, f"saved_models/best_loan_pipeline.pkl")
print(f"\nSelected Best Model: {best_model_name} (F1 Score: {best_score:.4f})")
print(f"Best pipeline saved as 'saved_models/best_loan_pipeline.pkl'")
