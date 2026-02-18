import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import webview
import joblib
import os
import datetime
import base64

# --- CONFIGURATION ---
MODEL_FILE = 'heart_model.joblib'
DATA_FILE = 'train.csv'

# --- 1. AI ENGINE ---
def initialize_engine():
    features = ['Age', 'Sex', 'Chest pain type', 'Cholesterol', 'BP', 'Max HR']
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE), features
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})
        X = df[features]
        y = df['Heart Disease']
        mdl = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
        mdl.fit(X, y)
        joblib.dump(mdl, MODEL_FILE)
        return mdl, features
    else:
        return None, features

model, feature_names = initialize_engine()

def get_icon_base64(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as image_file:
                return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"
    except: return None
    return None

icon_data = get_icon_base64("icon.png")

# --- 2. BACKEND API ---
class Api:
    def predict(self, data):
        try:
            age, sex = int(data['Age']), int(data['Sex'])
            cp, chol = int(data['CP']), float(data['Chol'])
            bp, hr = float(data['BP']), float(data['HR'])

            if model:
                df_input = pd.DataFrame([[age, sex, cp, chol, bp, hr]], columns=feature_names)
                prob = model.predict_proba(df_input)[0][1] * 100
            else:
                prob = 25.5 
            
            risk = round(prob, 1)
            med_tips = []
            food_tips = ["Eat fresh fruits and vegetables every day."]
            
            if bp > 140:
                med_tips.append("Your Blood Pressure is high. Please see a doctor.")
                food_tips.append("Use much less salt in your food.")
            if chol > 240:
                med_tips.append("Your Cholesterol is high. A doctor can help lower it.")
                food_tips.append("Avoid fried foods. Eat more oats and beans.")
            
            if risk > 70:
                status, color = "HIGH RISK", "#ff4757"
                med_tips.append("CRITICAL: High risk! See a heart specialist immediately.")
            elif risk > 30:
                status, color = "MEDIUM RISK", "#ffa502"
                med_tips.append("CAUTION: Medium risk detected. Watch your health closely.")
            else:
                status, color = "HEALTHY", "#2ed573"
                med_tips.append("Your heart looks healthy! Keep up your good lifestyle.")

            return {
                "risk": risk, "color": color, "status": status,
                "medical": med_tips, "food": food_tips,
                "timestamp": datetime.datetime.now().strftime("%I:%M %p")
            }
        except Exception as e: return {"error": str(e)}

# --- 3. UI DEFINITION ---
html_ui = f"""
<!DOCTYPE html>
<html>
<head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
    
    <style>
        :root {{
            --accent: #ff4757;
            --glass: rgba(255, 255, 255, 0.8);
        }}

        body {{ 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Plus Jakarta Sans', sans-serif; 
            height: 100vh; overflow: hidden; color: #2d3436;
        }}

        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes slideInLeft {{
            from {{ opacity: 0; transform: translateX(-30px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}

        .sidebar {{ 
            background: var(--glass); backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255,255,255,0.3);
            height: 100vh; padding: 30px; overflow-y: auto;
            animation: slideInLeft 0.8s ease-out;
        }}

        .dashboard {{ height: 100vh; padding: 40px; overflow-y: auto; }}

        .brand-box {{ 
            display: flex; align-items: center; gap: 12px; margin-bottom: 30px; 
        }}
        .brand-title {{ font-weight: 800; font-size: 1.4rem; color: var(--accent); letter-spacing: -1px; }}

        .form-label {{ font-size: 0.7rem; font-weight: 800; color: #636e72; text-transform: uppercase; margin-top: 15px; }}
        .form-control, .form-select {{ 
            border-radius: 12px; border: 1px solid rgba(0,0,0,0.05); padding: 12px; 
            transition: 0.3s; font-size: 0.9rem;
        }}
        .form-control:focus {{ box-shadow: 0 0 0 4px rgba(255, 71, 87, 0.1); border-color: var(--accent); }}

        .btn-predict {{ 
            background: var(--accent); color: white; border: none; padding: 16px; 
            border-radius: 15px; width: 100%; font-weight: 800; margin-top: 25px;
            box-shadow: 0 10px 20px rgba(255, 71, 87, 0.2); transition: 0.3s;
        }}
        .btn-predict:hover {{ transform: translateY(-2px); box-shadow: 0 15px 25px rgba(255, 71, 87, 0.3); }}

        .result-card {{ 
            background: var(--glass); backdrop-filter: blur(15px);
            border-radius: 30px; padding: 35px; border: 1px solid rgba(255,255,255,0.4);
            box-shadow: 0 20px 40px rgba(0,0,0,0.05);
            animation: fadeInUp 0.8s ease-out;
        }}

        .circle-container {{
            width: 220px; height: 220px; border-radius: 50%; border: 12px solid #eee;
            margin: 20px auto; display: flex; flex-direction: column; align-items: center; 
            justify-content: center; transition: 1s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        #score {{ font-size: 3.5rem; font-weight: 800; line-height: 1; }}
        #label {{ font-weight: 800; font-size: 0.9rem; text-transform: uppercase; margin-top: 5px; }}

        .tip-box {{ 
            background: white; border-radius: 15px; padding: 15px; margin-bottom: 12px;
            display: flex; align-items: center; gap: 15px; transition: 0.3s;
            border: 1px solid rgba(0,0,0,0.02);
        }}
        .tip-box:hover {{ transform: scale(1.02); box-shadow: 0 10px 20px rgba(0,0,0,0.03); }}
        .tip-icon {{ font-size: 1.2rem; min-width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }}
        
        .section-title {{ font-weight: 800; font-size: 0.75rem; color: #b2bec3; text-transform: uppercase; margin-bottom: 15px; display: block; }}
    </style>
</head>
<body>
    <div class="container-fluid p-0">
        <div class="row g-0">
            <div class="col-md-3 sidebar">
                <div class="brand-box">
                    {"<img src='" + icon_data + "' style='width:35px'>" if icon_data else "<i class='fa-solid fa-heart-pulse fa-2x text-danger'></i>"}
                    <div class="brand-title">HEARTGUARD</div>
                </div>

                <label class="form-label">Patient Age</label>
                <select id="Age" class="form-select"></select>

                <label class="form-label">Biological Sex</label>
                <select id="Sex" class="form-select"><option value="1">Male</option><option value="0">Female</option></select>

                <label class="form-label">Chest Pain Level</label>
                <select id="CP" class="form-select">
                    <option value="4">No Pain</option><option value="1">Mild</option>
                    <option value="2">Moderate</option><option value="3">Severe</option>
                </select>

                <label class="form-label">Cholesterol (mg/dL)</label>
                <input type="number" id="Chol" class="form-control" value="239" min="0" oninput="validity.valid||(value='0');">

                <label class="form-label">Blood Pressure (mmHg)</label>
                <input type="number" id="BP" class="form-control" value="130" min="0" oninput="validity.valid||(value='0');">

                <label class="form-label">Max Heart Rate</label>
                <input type="number" id="HR" class="form-control" value="150" min="0" oninput="validity.valid||(value='0');">

                <button class="btn-predict" onclick="analyze()">ANALYZE NOW</button>
            </div>

            <div class="col-md-9 dashboard">
                <div id="idle" class="text-center mt-5 pt-5">
                    <i class="fa-solid fa-wand-magic-sparkles fa-5x mb-4" style="color:rgba(0,0,0,0.1)"></i>
                    <h2 class="fw-300">Welcome to AI Health</h2>
                    <p class="text-muted">Enter patient vitals to generate a heart risk assessment.</p>
                </div>

                <div id="active" style="display:none;">
                    <div class="row g-4">
                        <div class="col-md-5">
                            <div class="result-card text-center">
                                <span class="section-title">Analysis Result</span>
                                <div id="ring" class="circle-container">
                                    <div id="score">0%</div>
                                    <div id="label">...</div>
                                </div>
                                <p class="small text-muted mb-0">Updated: <span id="ts"></span></p>
                            </div>
                        </div>
                        <div class="col-md-7">
                            <div class="result-card">
                                <span class="section-title">Medical Findings</span>
                                <div id="med-list"></div>
                                <span class="section-title mt-4">Lifestyle Plan</span>
                                <div id="food-list"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const ageBox = document.getElementById('Age');
        for(let i=1; i<=100; i++) {{
            let o = new Option(i, i);
            if(i == 45) o.selected = true;
            ageBox.add(o);
        }}

        function animateValue(obj, start, end, duration) {{
            let startTimestamp = null;
            const step = (timestamp) => {{
                if (!startTimestamp) startTimestamp = timestamp;
                const progress = Math.min((timestamp - startTimestamp) / duration, 1);
                obj.innerHTML = (progress * (end - start) + start).toFixed(1) + "%";
                if (progress < 1) {{
                    window.requestAnimationFrame(step);
                }}
            }};
            window.requestAnimationFrame(step);
        }}

        async function analyze() {{
            const inputs = {{
                Age: document.getElementById('Age').value,
                Sex: document.getElementById('Sex').value,
                CP: document.getElementById('CP').value,
                Chol: document.getElementById('Chol').value,
                BP: document.getElementById('BP').value,
                HR: document.getElementById('HR').value
            }};

            const res = await pywebview.api.predict(inputs);
            document.getElementById('idle').style.display = 'none';
            document.getElementById('active').style.display = 'block';

            animateValue(document.getElementById('score'), 0, res.risk, 1200);
            
            const ring = document.getElementById('ring');
            const scoreTxt = document.getElementById('score');
            const labelTxt = document.getElementById('label');

            scoreTxt.style.color = res.color;
            labelTxt.innerText = res.status;
            labelTxt.style.color = res.color;
            ring.style.borderColor = res.color;
            document.getElementById('ts').innerText = res.timestamp;

            document.getElementById('med-list').innerHTML = res.medical.map(m => 
                `<div class="tip-box">
                    <div class="tip-icon" style="background:${{res.color}}15; color:${{res.color}}">
                        <i class="fa-solid fa-stethoscope"></i>
                    </div>
                    <span>${{m}}</span>
                </div>`).join('');
            
            document.getElementById('food-list').innerHTML = res.food.map(f => 
                `<div class="tip-box">
                    <div class="tip-icon" style="background:#2ecc7115; color:#2ecc71">
                        <i class="fa-solid fa-leaf"></i>
                    </div>
                    <span>${{f}}</span>
                </div>`).join('');
        }}
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    window = webview.create_window("HeartGuard AI", html=html_ui, js_api=Api(), width=1300, height=900)
    webview.start()
