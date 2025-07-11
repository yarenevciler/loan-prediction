# 🧠 MLOps Loan Approval Prediction System

This project demonstrates a complete end-to-end **MLOps pipeline** for predicting loan approvals. It combines model training, API serving, web-based user interaction, and experiment tracking – all containerized using Docker Compose.

---

## 📦 Key Components

| Component         | Description                                              |
|-------------------|----------------------------------------------------------|
| **FastAPI**       | Serves the trained ML model through a REST API           |
| **Streamlit**     | Provides an interactive frontend to collect input        |
| **MLflow**        | Tracks experiments, metrics, and model versions          |
| **MySQL**         | Stores experiment metadata for MLflow                    |
| **MinIO**         | Acts as an S3-compatible storage for model artifacts     |
| **Docker Compose**| Orchestrates and manages all services                    |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/loan-mlops-project.git
cd loan-mlops-project
```

---

### 2. Build Docker Images

```bash
docker-compose build --no-cache
```

---

### 3. Start All Services

```bash
docker-compose up -d
```

---

### 4. Create the MySQL Database

After the containers are running, create the `mlops` database.

#### Option 1: Interactive

```bash
docker exec -it mysql mysql -u root -p
```

Then inside the MySQL shell:

```sql
CREATE DATABASE mlops;
```

#### Option 2: One-liner

```bash
docker exec -it mysql mysql -u root -p -e "CREATE DATABASE mlops;"
```

---

### 5. Train and Register the Model

Use the following command to train the model and log it to MLflow:

```bash
python train.py
```

This will:
- ✅ Train a scikit-learn model  
- ✅ Log metrics and model to MLflow  
- ✅ Store artifacts in MinIO  
- ✅ Register the model for serving  

---

## 🌐 Interfaces

| Interface         | URL                        |
|-------------------|----------------------------|
| **Streamlit App** | http://localhost:8502      |
| **FastAPI Docs**  | http://localhost:8001/docs |
| **MLflow UI**     | http://localhost:5001      |
| **MinIO Console** | http://localhost:9001      |

---

## 🧪 Making Predictions

### ▶️ Using Streamlit

1. Visit: [http://localhost:8502](http://localhost:8502)  
2. Enter loan applicant information  
3. Click **"Predict"** to view the result  

---

### ▶️ Using FastAPI Swagger UI

1. Visit: [http://localhost:8001/docs](http://localhost:8001/docs)  
2. Use the `/predict` endpoint with the following JSON:

```json
{
  "Gender": 1,
  "Married": 1,
  "Dependents": 0,
  "Education": 1,
  "Self_Employed": 0,
  "LoanAmount_log": 4.5,
  "Loan_Amount_Term": 360.0,
  "Credit_History": 1.0,
  "TotalIncome_log": 10.5,
  "Property_Area": "Urban"
}
```

---

## 📊 View Stored Predictions (Optional)

If predictions are stored in MySQL, you can query them like this:

```bash
docker exec -it mysql mysql -u mlops_user -p -e "SELECT * FROM mlops.loanprediction;"
```

---

## 📁 Project Structure

\`\`\`bash
.
├── docker-compose.yml
├── .env
├── train.py
├── main.py
├── streamlit_app/
│   ├── app.py
│   └── requirements.txt
├── mlflow/
├── sql_scripts/
│   └── mysql_init.sql
├── wait-for-it.sh
└── README.md
\`\`\`

---

## ⚙️ Tech Stack

- Python  
- FastAPI  
- Streamlit  
- MLflow  
- MySQL  
- MinIO  
- Docker Compose  

---

## 👩‍💻 Author

**Yaren Evciler**  
AI Engineer | MLOps Enthusiast

---

## 📄 License

This project is for educational and demonstration purposes.
