import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import webview
import joblib
import os
import datetime
import random

# --- CONFIGURATION ---
MODEL_FILE = 'heart_model.joblib'
DATA_FILE = 'Train.xlsx - Sheet1.csv'  # Linked to your uploaded file

# --- 1. AI ENGINE ---
def initialize_engine():
    # Application feature names used for prediction
    features = ['Age', 'Sex', 'Chest pain type', 'Cholesterol', 'BP', 'Max HR']
    
    # Try to load existing model
    if os.path.exists(MODEL_FILE):
        try: return joblib.load(MODEL_FILE), features
        except: pass

    # Train model if data file exists
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            
            # Map dataset columns to application feature names
            # age -> Age, sex -> Sex, cp -> Chest pain type, chol -> Cholesterol, trestbps -> BP, thalach -> Max HR
            mapping = {
                'age': 'Age',
                'sex': 'Sex',
                'cp': 'Chest pain type',
                'chol': 'Cholesterol',
                'trestbps': 'BP',
                'thalach': 'Max HR'
            }
            
            # Prepare Features (X)
            X = df[list(mapping.keys())].rename(columns=mapping)
            
            # Prepare Target (y): Convert 'num' (0-4) to binary (0=Healthy, 1=Presence)
            y = df['num'].apply(lambda x: 1 if x > 0 else 0)
            
            # Initialize and Train Random Forest
            mdl = RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42)
            mdl.fit(X, y)
            
            # Save for faster future startups
            joblib.dump(mdl, MODEL_FILE)
            return mdl, features
        except Exception as e:
            print(f"Training Error: {e}")
            return None, features
            
    return None, features

model, feature_names = initialize_engine()

# --- 2. BACKEND API ---
class Api:
    def predict(self, data):
        try:
            age, sex = int(data['Age']), int(data['Sex'])
            cp, chol = int(data['CP']), float(data['Chol'])
            bp, hr = float(data['BP']), float(data['HR'])

            # 1. Evaluate Individual Levels
            bp_lvl = "Normal"
            if bp >= 140: bp_lvl = "High"
            elif bp < 90: bp_lvl = "Low"

            hr_lvl = "Normal"
            if hr >= 100: hr_lvl = "High"
            elif hr < 50: hr_lvl = "Low"

            chol_lvl = "Normal"
            if chol >= 240: chol_lvl = "High"

            # 2. Emergency Check
            is_emergency = False
            emergency_msg = ""
            if bp > 180 or hr > 160 or hr < 40 or cp == 4:
                is_emergency = True
                emergency_msg = "CRITICAL: Your vitals are at a dangerous level. Please seek medical help immediately."

            # 3. AI Prediction
            if model:
                # Ensure input order matches training features
                df_input = pd.DataFrame([[age, sex, cp, chol, bp, hr]], columns=feature_names)
                prob = model.predict_proba(df_input)[0][1] * 100
            else:
                # Fallback simple logic if model isn't trained
                prob = (age*0.3) + (chol/10) + (bp/5) + (cp*10) - (hr/10)

            risk = max(2, min(98, round(prob, 1)))

            # 4. Dynamic Tips
            health_tips = [
                "Walking 30 mins a day strengthens the heart muscle.",
                "Reduce salt intake to lower high blood pressure.",
                "Eat more fiber (oats, beans) to lower bad cholesterol.",
                "Avoid smoking to keep your arteries flexible.",
                "Manage stress through deep breathing or meditation.",
                "Omega-3 in fish is like 'oil' for your heart's health."
            ]
            
            emergency_tips = [
                "Sit down and try to remain calm.",
                "Loosen tight clothing to breathe easier.",
                "Call your local emergency number.",
                "Do not try to drive yourself to the hospital."
            ]

            # 5. Result Status
            if is_emergency:
                status, color = "EMERGENCY", "#d63031"
                display_msg = emergency_msg
                tips_to_show = emergency_tips
            elif risk > 40:
                status, color = "HIGH RISK", "#e17055"
                display_msg = "Your heart needs more care. Talk to a doctor about these numbers."
                tips_to_show = random.sample(health_tips, 3)
            else:
                status, color = "HEALTHY", "#00b894"
                display_msg = "Great job! Your heart vitals look good. Keep your healthy habits."
                tips_to_show = random.sample(health_tips, 3)

            return {
                "risk": risk, "status": status, "color": color, "msg": display_msg,
                "bp_val": f"{bp} ({bp_lvl})", "hr_val": f"{hr} ({hr_lvl})", "chol_val": f"{chol} ({chol_lvl})",
                "tips": tips_to_show, "is_emergency": is_emergency,
                "timestamp": datetime.datetime.now().strftime("%I:%M %p")
            }
        except Exception as e:
            return {"error": str(e)}

# --- 3. UI DEFINITION ---
html_ui = """
<!DOCTYPE html>
<html>
<head>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #0984e3; --bg: #f8f9fa; --card: #ffffff; --text: #2d3436; }
        body { margin: 0; font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; }
        
        .sidebar { width: 380px; background: var(--card); border-right: 1px solid #ddd; padding: 30px; overflow-y: auto; }
        .main { flex: 1; padding: 40px; overflow-y: auto; display: flex; flex-direction: column; align-items: center; }
        
        .input-box { margin-bottom: 15px; display: flex; flex-direction: column; }
        label { font-size: 0.85rem; font-weight: 700; margin-bottom: 5px; color: #636e72; }
        input, select { padding: 12px; border: 1.5px solid #dfe6e9; border-radius: 8px; font-size: 1rem; }
        
        .btn { background: var(--primary); color: white; border: none; padding: 15px; border-radius: 8px; font-weight: 700; width: 100%; cursor: pointer; margin-top: 10px; }
        
        .res-card { background: white; border-radius: 20px; padding: 30px; width: 100%; max-width: 700px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); text-align: center; }
        .score-box { font-size: 3.5rem; font-weight: 800; margin: 10px 0; }
        
        .vitals-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 20px 0; }
        .vital { background: #f1f2f6; padding: 15px; border-radius: 12px; font-size: 0.9rem; }
        .vital b { display: block; font-size: 1.1rem; color: var(--primary); }

        .tips-box { width: 100%; max-width: 700px; margin-top: 20px; background: #fff; padding: 25px; border-radius: 20px; border-left: 8px solid var(--primary); }
        .emergency-style { background: #fff5f5; border-color: #ff7675; }
        .emergency-text { color: #d63031; font-weight: 700; }
        
        ul { text-align: left; padding-left: 20px; line-height: 1.6; }
    </style>
</head>
<body>

<div class="sidebar">
    <h2 style="color:#d63031"><i class="fa-solid fa-heart-pulse"></i> HeartCheck</h2>
    <div class="input-box"><label>Age</label><input type="number" id="Age" value="45"></div>
    <div class="input-box"><label>Sex</label><select id="Sex"><option value="1">Male</option><option value="0">Female</option></select></div>
    <div class="input-box"><label>Chest Pain</label><select id="CP"><option value="1">None</option><option value="2">Mild</option><option value="3">Moderate</option><option value="4">Severe/Sharp</option></select></div>
    <div class="input-box"><label>Cholesterol</label><input type="number" id="Chol" value="200"></div>
    <div class="input-box"><label>Blood Pressure</label><input type="number" id="BP" value="120"></div>
    <div class="input-box"><label>Heart Rate</label><input type="number" id="HR" value="75"></div>
    <button class="btn" onclick="analyze()">RUN ANALYSIS</button>
</div>

<div class="main">
    <div id="results" style="display:none; width:100%; max-width:700px">
        <div class="res-card" id="card">
            <div id="status-label" style="font-weight:700; letter-spacing:1px">STATUS</div>
            <div class="score-box" id="risk-score">0%</div>
            <p id="msg-text"></p>
            
            <div class="vitals-row">
                <div class="vital"><span>Pressure</span><b id="v-bp">--</b></div>
                <div class="vital"><span>Heart Rate</span><b id="v-hr">--</b></div>
                <div class="vital"><span>Cholesterol</span><b id="v-chol">--</b></div>
            </div>
        </div>

        <div class="tips-box" id="tips-box">
            <h3 id="tips-title"><i class="fa-solid fa-user-doctor"></i> Recommended Actions:</h3>
            <ul id="tips-list"></ul>
        </div>
    </div>
</div>

<script>
async function analyze(){
    const data = {
        Age: document.getElementById('Age').value, 
        Sex: document.getElementById('Sex').value, 
        CP: document.getElementById('CP').value,
        Chol: document.getElementById('Chol').value, 
        BP: document.getElementById('BP').value, 
        HR: document.getElementById('HR').value
    };
    const res = await pywebview.api.predict(data);
    
    if (res.error) {
        alert("Error: " + res.error);
        return;
    }

    document.getElementById('results').style.display = 'block';
    
    // Set UI Colors
    document.getElementById('risk-score').innerText = res.risk + "%";
    document.getElementById('risk-score').style.color = res.color;
    document.getElementById('status-label').innerText = res.status;
    document.getElementById('status-label').style.color = res.color;
    document.getElementById('msg-text').innerText = res.msg;
    
    // Set Vital Texts
    document.getElementById('v-bp').innerText = res.bp_val;
    document.getElementById('v-hr').innerText = res.hr_val;
    document.getElementById('v-chol').innerText = res.chol_val;

    // Handle Emergency UI
    const tBox = document.getElementById('tips-box');
    const tTitle = document.getElementById('tips-title');
    if(res.is_emergency){
        tBox.className = "tips-box emergency-style";
        tTitle.className = "emergency-text";
        tTitle.innerHTML = "<i class='fa-solid fa-triangle-exclamation'></i> EMERGENCY STEPS:";
    } else {
        tBox.className = "tips-box";
        tTitle.className = "";
        tTitle.innerHTML = "<i class='fa-solid fa-user-doctor'></i> Healthy Heart Tips:";
    }

    // Build Tips List
    const list = document.getElementById('tips-list');
    list.innerHTML = "";
    res.tips.forEach(tip => {
        let li = document.createElement('li');
        li.innerText = tip;
        list.appendChild(li);
    });
}
</script>

</body>
</html>
"""

if __name__ == '__main__':
    webview.create_window("HeartCheck AI Pro", html=html_ui, js_api=Api(), width=1200, height=850)
    webview.start()
