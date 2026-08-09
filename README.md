# AI_Powered_Customer_Churn_Prediction

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a telecom customer is likely to **churn (leave the service)** or **continue using the service**. The model analyzes customer demographics, subscription details, billing information, and service usage to assist businesses in identifying customers at risk of leaving.

This project includes data preprocessing, feature engineering, model training, model evaluation, and a Flask web application for real-time predictions.

---

## 🚀 Features

- Data Cleaning and Preprocessing
- Handling Missing Values
- Feature Engineering
- Variable Transformation
- One-Hot Encoding and Ordinal Encoding Techniques
- Data Balancing
- Feature Scaling using StandardScaler
- Trained on multiple models
- Logistic Regression Model
- Model Serialization using Pickle
- Flask Web Application
- User-Friendly Prediction Interface

---

## 📂 Dataset

The project uses the **Telco Customer Churn Dataset**.

### Dataset Features

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

### Target Variable

- **Churn**
  - 0 → Customer Will Stay
  - 1 → Customer Will Not stay

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- Pickle

---

## 📈 Data Preprocessing

The following preprocessing techniques were applied:

- Missing Value Handling
- Constant Value Imputation
- Yeo-Johnson Transformation
- One-Hot Encoding
- Feature Scaling using StandardScaler
- Data Balancing
- Train-Test Split

---

## 🤖 Machine Learning Model

The project uses the **Logistic Regression** algorithm for customer churn prediction.

### Model Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Data Transformation
5. Data Balancing
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Model Saving using Pickle
10. Deployment using Flask

---

## 📊 Model Evaluation Metrics

The model was evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
- ROC-Curve 
- ROC-AUC Score

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
│
├── .venv/
│
├── log/
│
├── templates/
│   └── index.html
│
├── all_models.py
├── app.py
├── constant_technique.py
├── Customer_Churn.pkl
├── Customer_Information.csv
├── feature_selection.py
├── log_code.py
├── main.py
├── ROC_Curve.png
├── Standard_scaler.pkl
├── var_outlier.py
├── README.md
└── requirements.txt
```

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/Tejasri0/AI_Powered_Customer_Retention_System.git
```

### Move into Project Directory

```bash
cd customer-churn-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Flask Application

```bash
python ./app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📌 Prediction Output

The application predicts one of the following:

### ✅ Customer Will Stay

or

### ⚠️ Customer Will Not Stay

---

## 💡 Future Enhancements

- Deploy using Render or Heroku
- Interactive Dashboard
- Probability Score Display
- Model Explainability using SHAP
- Database Integration
- User Authentication
- REST API Support

---

## 📚 Requirements

```
Flask
numpy
pandas
scikit-learn
pickle
```

---

## 👩‍💻 Author

**Kambhampalli Tejasri**
