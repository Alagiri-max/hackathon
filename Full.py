import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import webview
import joblib
import os
import datetime
import base64
import mysql.connector

# ==============================
# CONFIGURATION
# ==============================
MODEL_FILE = 'heart_model.joblib'
DATA_FILE = 'train.csv'

# ==============================
# DATABASE CONNECTION
# ==============================
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # put your mysql password
        database="heartguard"
    )
    cursor = db.cursor()
    print("✅ Database Connected")
except Exception as e:
    print("❌ Database Error:", e)
    db = None

# ==============================
# AI MODEL ENGINE
# ==============================
def initialize_engine():
    features = ['Age', 'Sex', 'Chest pain type', 'Cholesterol', 'BP', 'Max HR']
    
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE), features

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})
        X = df[features]
        y = df['Heart Disease']

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            n_jobs=-1,
            random_state=42
        )

        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        return model, features

    return None, features

model, feature_names = initialize_engine()

# ==============================
# API CLASS
# ==============================
class Api:

    def predict(self, data):
        try:
            age = int(data['Age'])
            sex = int(data['Sex'])
            cp = int(data['CP'])
            chol = float(data['Chol'])
            bp = float(data['BP'])
            hr = float(data['HR'])

            # Prediction
            if model:
                df_input = pd.DataFrame(
                    [[age, sex, cp, chol, bp, hr]],
                    columns=feature_names
                )
                prob = model.predict_proba(df_input)[0][1] * 100
            else:
                prob = 25.5  # fallback

            risk = round(prob, 1)

            # Risk logic
            med_tips = []
            food_tips = ["Eat fresh fruits and vegetables daily."]

            if bp > 140:
                med_tips.append("High Blood Pressure detected.")
                food_tips.append("Reduce salt intake.")

            if chol > 240:
                med_tips.append("High Cholesterol detected.")
                food_tips.append("Avoid fried food. Eat oats and beans.")

            if risk > 70:
                status = "HIGH RISK"
                color = "#ff4757"
                med_tips.append("CRITICAL: Consult cardiologist immediately.")
            elif risk > 30:
                status = "MEDIUM RISK"
                color = "#ffa502"
                med_tips.append("CAUTION: Monitor health closely.")
            else:
                status = "HEALTHY"
                color = "#2ed573"
                med_tips.append("Heart condition looks good.")

            # ==============================
            # SAVE TO DATABASE
            # ==============================
            if db:
                try:
                    sql = """
                        INSERT INTO predictions 
                        (age, sex, chest_pain, cholesterol, bp, max_hr, risk, status, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (
                        age, sex, cp, chol, bp, hr,
                        risk, status,
                        datetime.datetime.now()
                    )
                    cursor.execute(sql, values)
                    db.commit()
                except Exception as db_error:
                    print("DB Insert Error:", db_error)

            return {
                "risk": risk,
                "color": color,
                "status": status,
                "medical": med_tips,
                "food": food_tips,
                "timestamp": datetime.datetime.now().strftime("%I:%M %p")
            }

        except Exception as e:
            return {"error": str(e)}

# ==============================
# SIMPLE UI (Clean Version)
# ==============================
html_ui = """
<!DOCTYPE html>
<html>
<head>
<title>HeartGuard AI</title>
<style>
body {
    font-family: Arial;
    background: linear-gradient(135deg,#f5f7fa,#c3cfe2);
    text-align:center;
    padding-top:50px;
}
input, select {
    padding:10px;
    margin:8px;
    width:200px;
}
button {
    padding:12px 20px;
    background:#ff4757;
    color:white;
    border:none;
    cursor:pointer;
}
#result {
    margin-top:30px;
    font-size:20px;
    font-weight:bold;
}
</style>
</head>
<body>

<h2>HeartGuard AI Predictor</h2>

<select id="Age"></select><br>
<select id="Sex">
<option value="1">Male</option>
<option value="0">Female</option>
</select><br>

<input type="number" id="CP" placeholder="Chest Pain (1-4)" value="2"><br>
<input type="number" id="Chol" placeholder="Cholesterol" value="239"><br>
<input type="number" id="BP" placeholder="Blood Pressure" value="130"><br>
<input type="number" id="HR" placeholder="Max Heart Rate" value="150"><br>

<button onclick="predict()">ANALYZE</button>

<div id="result"></div>

<script>
for(let i=1;i<=100;i++){
    let o=new Option(i,i);
    if(i==45)o.selected=true;
    Age.add(o);
}

async function predict(){
    let data={
        Age:Age.value,
        Sex:Sex.value,
        CP:CP.value,
        Chol:Chol.value,
        BP:BP.value,
        HR:HR.value
    };

    let res=await pywebview.api.predict(data);

    document.getElementById("result").innerHTML=
        "<span style='color:"+res.color+"'>"+
        res.status+" - "+res.risk+"%</span>";
}
</script>

</body>
</html>
"""

# ==============================
# START APP
# ==============================
if __name__ == '__main__':
    window = webview.create_window(
        "HeartGuard AI",
        html=html_ui,
        js_api=Api(),
        width=800,
        height=700
    )
    webview.start()
