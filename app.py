from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load model and scaler
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("standar_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    sim = None

    if request.method == "POST":
        try:
            # Numeric Inputs
            MonthlyCharges = float(request.form['MonthlyCharges'])
            TotalCharges = float(request.form['TotalCharges'])
            tenure = int(request.form['tenure'])

            # Binary Inputs
            gender = int(request.form['gender'])
            Partner = int(request.form['Partner'])
            Dependents = int(request.form['Dependents'])
            PhoneService = int(request.form['PhoneService'])
            PaperlessBilling = int(request.form['PaperlessBilling'])

            # Categorical Inputs
            MultipleLines = request.form['MultipleLines']
            InternetService = request.form['InternetService']
            OnlineSecurity = request.form['OnlineSecurity']
            OnlineBackup = request.form['OnlineBackup']
            DeviceProtection = request.form['DeviceProtection']
            TechSupport = request.form['TechSupport']
            SIM_provider = request.form['SIM_provider']

            sim = SIM_provider  # send to HTML

            # Feature vector
            features = np.zeros(29)

            # Numeric
            features[0] = MonthlyCharges
            features[1] = TotalCharges
            features[2] = tenure

            # Binary
            features[3] = gender
            features[4] = Partner
            features[5] = Dependents
            features[6] = PhoneService
            features[7] = PaperlessBilling

            # One-hot encoding

            # MultipleLines
            if MultipleLines == "No phone service":
                features[8] = 1
            elif MultipleLines == "Yes":
                features[9] = 1

            # InternetService
            if InternetService == "DSL":
                features[10] = 1
            elif InternetService == "Fiber optic":
                features[11] = 1
            elif InternetService == "No":
                features[12] = 1

            # OnlineSecurity
            if OnlineSecurity == "No internet service":
                features[13] = 1
            elif OnlineSecurity == "Yes":
                features[14] = 1

            # OnlineBackup
            if OnlineBackup == "No internet service":
                features[15] = 1
            elif OnlineBackup == "Yes":
                features[16] = 1

            # DeviceProtection
            if DeviceProtection == "No internet service":
                features[17] = 1
            elif DeviceProtection == "Yes":
                features[18] = 1

            # TechSupport
            if TechSupport == "No internet service":
                features[19] = 1
            elif TechSupport == "Yes":
                features[20] = 1

            # SIM Provider
            if SIM_provider == "Jio":
                features[21] = 1
            elif SIM_provider == "Airtel":
                features[22] = 1
            elif SIM_provider == "Vodafone":
                features[23] = 1
            elif SIM_provider == "BSNL":
                features[24] = 1

            # Scaling
            features_scaled = scaler.transform([features])

            # Prediction
            prediction = model.predict(features_scaled)[0]

            result = "Customer will stay" if prediction == 0 else "Customer will churn"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, sim=sim)


if __name__ == "__main__":
    app.run(debug=True)