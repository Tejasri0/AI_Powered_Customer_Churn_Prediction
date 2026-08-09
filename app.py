from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("Customer_Churn.pkl", "rb") as f:
    model = pickle.load(f)

# Load Scaler
with open("Standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    pred = None
    risk_percent = None
    error = None

    if request.method == "POST":

        try:

            # ===========================
            # Numerical Features
            # ===========================

            SeniorCitizen = float(request.form["SeniorCitizen"])
            tenure = float(request.form["tenure_yeo"])
            monthly = float(request.form["MonthlyCharges_yeo"])
            total = float(request.form["TotalCharges_const_yeo"])

            # ===========================
            # Binary Features
            # ===========================

            gender_Male = float(request.form["gender_Male"])
            Partner_Yes = float(request.form["Partner_Yes"])
            Dependents_Yes = float(request.form["Dependents_Yes"])
            PhoneService_Yes = float(request.form["PhoneService_Yes"])
            PaperlessBilling_Yes = float(request.form["PaperlessBilling_Yes"])

            # ===========================
            # Multiple Lines
            # ===========================

            multiple = request.form["MultipleLines"]

            MultipleLines_No_phone_service = 1 if multiple == "No phone service" else 0
            MultipleLines_Yes = 1 if multiple == "Yes" else 0

            # ===========================
            # Internet Service
            # ===========================

            internet = request.form["InternetService"]

            InternetService_Fiber_optic = 1 if internet == "Fiber optic" else 0
            InternetService_No = 1 if internet == "No" else 0

            # ===========================
            # Online Security
            # ===========================

            security = request.form["OnlineSecurity"]

            OnlineSecurity_No_internet = 1 if security == "No internet service" else 0
            OnlineSecurity_Yes = 1 if security == "Yes" else 0

            # ===========================
            # Online Backup
            # ===========================

            backup = request.form["OnlineBackup"]

            OnlineBackup_No_internet = 1 if backup == "No internet service" else 0
            OnlineBackup_Yes = 1 if backup == "Yes" else 0

            # ===========================
            # Device Protection
            # ===========================

            device = request.form["DeviceProtection"]

            DeviceProtection_No_internet = 1 if device == "No internet service" else 0
            DeviceProtection_Yes = 1 if device == "Yes" else 0

            # ===========================
            # Tech Support
            # ===========================

            tech = request.form["TechSupport"]

            TechSupport_No_internet = 1 if tech == "No internet service" else 0
            TechSupport_Yes = 1 if tech == "Yes" else 0

            # ===========================
            # Streaming TV
            # ===========================

            tv = request.form["StreamingTV"]

            StreamingTV_No_internet = 1 if tv == "No internet service" else 0
            StreamingTV_Yes = 1 if tv == "Yes" else 0

            # ===========================
            # Streaming Movies
            # ===========================

            movie = request.form["StreamingMovies"]

            StreamingMovies_No_internet = 1 if movie == "No internet service" else 0
            StreamingMovies_Yes = 1 if movie == "Yes" else 0

            # ===========================
            # Payment Method
            # ===========================

            payment = request.form["PaymentMethod"]

            Payment_Credit = 1 if payment == "Credit card (automatic)" else 0
            Payment_Electronic = 1 if payment == "Electronic check" else 0
            Payment_Mailed = 1 if payment == "Mailed check" else 0

            # Bank Transfer is the reference category

            # ===========================
            # SIM Type
            # ===========================

            sim = request.form["sim_type"]

            sim_bsnl = 1 if sim == "BSNL" else 0
            sim_jio = 1 if sim == "Jio" else 0
            sim_vodafone = 1 if sim == "Vodafone" else 0

            # Airtel is the reference category

            # ===========================
            # Contract
            # ===========================

            Contract = float(request.form["Contract_res"])

            # ===========================
            # Final Feature Vector
            # ===========================

            features = np.array([[
                SeniorCitizen,
                tenure,
                monthly,
                total,

                gender_Male,
                Partner_Yes,
                Dependents_Yes,
                PhoneService_Yes,

                MultipleLines_No_phone_service,
                MultipleLines_Yes,

                InternetService_Fiber_optic,
                InternetService_No,

                OnlineSecurity_No_internet,
                OnlineSecurity_Yes,

                OnlineBackup_No_internet,
                OnlineBackup_Yes,

                DeviceProtection_No_internet,
                DeviceProtection_Yes,

                TechSupport_No_internet,
                TechSupport_Yes,

                StreamingTV_No_internet,
                StreamingTV_Yes,

                StreamingMovies_No_internet,
                StreamingMovies_Yes,

                PaperlessBilling_Yes,

                Payment_Credit,
                Payment_Electronic,
                Payment_Mailed,

                sim_bsnl,
                sim_jio,
                sim_vodafone,

                Contract
            ]])

            # Scale Features
            features = scaler.transform(features)

            # Prediction
            pred = int(model.predict(features)[0])

            # Try to get a churn probability for the risk meter (falls back
            # gracefully if the model doesn't support predict_proba)
            try:
                risk_percent = round(float(model.predict_proba(features)[0][1]) * 100, 1)
            except AttributeError:
                risk_percent = None

            if pred == 1:
                prediction = "Customer Will not Stay"
            else:
                prediction = "Customer Will Stay"

        except Exception as e:
            error = f"Could not generate a prediction — {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        pred=pred,
        risk_percent=risk_percent,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
