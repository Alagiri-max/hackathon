import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import webview
import joblib
import os
import datetime
import base64

MODEL_FILE = 'heart_model.joblib'

# --- 1. AI ENGINE ---
def initialize_engine():
    features = ['Age', 'Sex', 'Chest pain type', 'Cholesterol', 'BP', 'Max HR']
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE), features
    
    data = {
        'Age': np.random.randint(30, 80, 1000), 'Sex': np.random.randint(0, 2, 1000),
        'Chest pain type': np.random.randint(1, 5, 1000), 'Cholesterol': np.random.randint(150, 400, 1000),
        'BP': np.random.randint(90, 180, 1000), 'Max HR': np.random.randint(100, 200, 1000),
        'Heart Disease': np.random.choice(['Absence', 'Presence'], 1000)
    }
    df = pd.DataFrame(data)
    mdl = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    mdl.fit(df[features], df['Heart Disease'].map({'Absence': 0, 'Presence': 1}))
    joblib.dump(mdl, MODEL_FILE)
    return mdl, features

model, feature_names = initialize_engine()

# Function to convert your icon to a string for CSS/HTML reliability
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

            df_input = pd.DataFrame([[age, sex, cp, chol, bp, hr]], columns=feature_names)
            prob = model.predict_proba(df_input)[0][1] * 100
            risk = round(prob, 2)
            
            medical_tips = []
            food_tips = ["Eat more green vegetables and fresh fruits every day."]
            
            if bp > 140:
                medical_tips.append("Your blood pressure is high. Talk to a doctor about medicine.")
                food_tips.append("Use much less salt in your food.")
            if chol > 240:
                medical_tips.append("Your cholesterol is high. A doctor can help lower it.")
                food_tips.append("Avoid fried foods. Eat oats and beans.")
            if risk > 50:
                medical_tips.append("URGENT: Please book a heart check-up immediately.")
                food_tips.append("Eat walnuts and fish instead of red meat.")
            
            if not medical_tips:
                medical_tips.append("Keep doing your yearly health check-ups.")

            return {
                "risk": risk,
                "color": "#e74c3c" if risk > 70 else "#f39c12" if risk > 30 else "#2ecc71",
                "status": "HIGH RISK" if risk > 70 else "MILD RISK" if risk > 30 else "HEALTHY",
                "medical": medical_tips,
                "food": food_tips,
                "timestamp": datetime.datetime.now().strftime("%I:%M %p")
            }
        except Exception as e: return {"error": str(e)}

# --- 3. PREMIUM UI ---
html_ui = f"""
<!DOCTYPE html>
<html>
<head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        body {{ background: #f0f2f5; font-family: 'Inter', sans-serif; color: #2d3436; height: 100vh; overflow: hidden; }}
        .sidebar {{ background: #ffffff; border-right: 1px solid #e1e4e8; height: 100vh; padding: 25px; overflow-y: auto; }}
        .dashboard {{ background: #f8f9fa; height: 100vh; padding: 30px; overflow-y: auto; }}
        
        .brand-section {{ background: #fff5f5; padding: 15px; border-radius: 12px; border: 1px solid #feb2b2; margin-bottom: 25px; display: flex; align-items: center; }}
        .brand-logo {{ width: 50px; height: 50px; margin-right: 15px; border-radius: 8px; object-fit: cover; }}
        .brand-title {{ font-weight: 800; font-size: 1.3rem; color: #c62828; line-height: 1; }}
        .brand-tagline {{ font-size: 0.65rem; color: #e53e3e; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }}

        .form-label {{ font-weight: 600; font-size: 0.7rem; color: #636e72; text-transform: uppercase; margin-top: 10px; }}
        .form-control, .form-select {{ border-radius: 8px; margin-bottom: 8px; font-size: 0.9rem; border: 1px solid #dee2e6; }}
        .btn-predict {{ background: #c62828; color: white; border: none; padding: 14px; border-radius: 10px; width: 100%; font-weight: 700; margin-top: 15px; }}
        
        .result-card {{ background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.03); margin-bottom: 20px; }}
        .risk-circle {{ width: 220px; height: 220px; border-radius: 50%; border: 15px solid #f1f2f6; margin: 10px auto; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
        #risk-val {{ font-size: 3rem; font-weight: 800; line-height: 1; }}
        
        .tip-box {{ background: #fcfcfc; border-left: 4px solid #c62828; padding: 12px; margin-bottom: 10px; border-radius: 6px; font-size: 0.9rem; display: flex; align-items: center; }}
        .tip-icon {{ margin-right: 10px; color: #c62828; font-size: 1.1rem; }}
        
        .section-title {{ font-weight: 800; font-size: 0.8rem; color: #adb5bd; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; }}
        .footer-tag {{ position: fixed; bottom: 10px; left: 10px; font-size: 0.65rem; color: #ccc; }}
    </style>
</head>
<body>
    <div class="container-fluid p-0">
        <div class="row g-0">
            <div class="col-md-3 sidebar">
                <div class="brand-section">
                    {"<img src='" + icon_data + "' class='brand-logo'>" if icon_data else "<i class='fa-solid fa-heart-pulse' style='font-size: 2.5rem; color: #c62828; margin-right: 15px;'></i>"}
                    <div>
                        <div class="brand-title">HEARTGUARD</div>
                        <div class="brand-tagline">AI Smart Diagnostic</div>
                    </div>
                </div>

                <label class="form-label"><i class="fa-solid fa-calendar-day me-1"></i> Patient Age</label>
                <select id="Age" class="form-select"></select>
                
                <label class="form-label"><i class="fa-solid fa-venus-mars me-1"></i> Sex</label>
                <select id="Sex" class="form-select"><option value="1">Male</option><option value="0">Female</option></select>
                
                <label class="form-label"><i class="fa-solid fa-stethoscope me-1"></i> Chest Pain</label>
                <select id="CP" class="form-select">
                    <option value="4" selected>No pain</option><option value="1">Occasional discomfort</option>
                    <option value="2">Mild pressure</option><option value="3">Strong pain</option></select>
                
                <label class="form-label"><i class="fa-solid fa-droplet me-1"></i> Cholesterol (mg/dL)</label>
                <input type="number" id="Chol" class="form-control" value="180">
                
                <label class="form-label"><i class="fa-solid fa-gauge-high me-1"></i> Blood Pressure (mmHg)</label>
                <input type="number" id="BP" class="form-control" value="115">
                
                <label class="form-label"><i class="fa-solid fa-heart-pulse me-1"></i> Heart Rate (bpm)</label>
                <input type="number" id="HR" class="form-control" value="75">
                
                <button class="btn-predict" onclick="run()">
                    <i class="fa-solid fa-magnifying-glass-chart me-2"></i> ANALYZE HEART
                </button>
            </div>
            <div class="col-md-9 dashboard">
                <div id="idle" class="text-center mt-5">
                    <i class="fa-solid fa-hospital-user fa-4x text-muted mb-3" style="opacity:0.3"></i>
                    <h2 class="text-muted">Enter Patient Information</h2>
                    <p class="text-muted">Use standard medical values to begin cardiac assessment.</p>
                </div>
                <div id="active" style="display:none;">
                    <div class="row">
                        <div class="col-md-5">
                            <div class="result-card text-center">
                                <div class="section-title">Risk Assessment</div>
                                <div id="ring" class="risk-circle">
                                    <span id="risk-val">0%</span>
                                    <span id="status-label" style="font-weight:700"></span>
                                </div>
                                <small class="text-muted"><i class="fa-regular fa-clock me-1"></i> Analyzed at: <span id="time"></span></small>
                            </div>
                        </div>
                        <div class="col-md-7">
                            <div class="result-card">
                                <div class="section-title"><i class="fa-solid fa-user-doctor me-2"></i>Doctor's Advice</div>
                                <div id="med-list"></div>
                                <div class="section-title mt-4"><i class="fa-solid fa-apple-whole me-2"></i>Healthy Eating Tips</div>
                                <div id="food-list"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const ageS = document.getElementById('Age');
        for(let i=20; i<=90; i++){{ let o=new Option(i+' Years', i); if(i==45)o.selected=true; ageS.add(o); }}

        async function run() {{
            const data = {{ Age: document.getElementById('Age').value, Sex: document.getElementById('Sex').value, CP: document.getElementById('CP').value,
                           Chol: document.getElementById('Chol').value, BP: document.getElementById('BP').value, HR: document.getElementById('HR').value }};
            const res = await pywebview.api.predict(data);
            document.getElementById('idle').style.display = 'none'; document.getElementById('active').style.display = 'block';
            
            document.getElementById('risk-val').innerText = res.risk + '%';
            document.getElementById('risk-val').style.color = res.color;
            document.getElementById('status-label').innerText = res.status;
            document.getElementById('status-label').style.color = res.color;
            document.getElementById('ring').style.borderColor = res.color;
            document.getElementById('time').innerText = res.timestamp;

            document.getElementById('med-list').innerHTML = res.medical.map(m => `
                <div class="tip-box">
                    <i class="fa-solid fa-circle-info tip-icon"></i>
                    <span>${{m}}</span>
                </div>`).join('');
            document.getElementById('food-list').innerHTML = res.food.map(f => `
                <div class="tip-box" style="border-left-color:#2ecc71">
                    <i class="fa-solid fa-leaf tip-icon" style="color:#2ecc71"></i>
                    <span>${{f}}</span>
                </div>`).join('');
        }}
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    # Stable launcher: No version-specific icon arguments to ensure 100% uptime
    window = webview.create_window(
        "HeartGuard AI - Professional Dashboard", 
        html=html_ui, 
        js_api=Api(), 
        width=1200, 
        height=800
    )
    webview.start()
