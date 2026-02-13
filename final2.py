import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import webview
import joblib
import os
import sys

# ============================================================
# PHASE 1: LOGIC CODE (AI Brain with Persistence)
# ============================================================
MODEL_PATH = 'heart_pro_model.pkl'
FEATURES = ['Age', 'Sex', 'Chest pain type', 'Cholesterol', 'BP', 'Max HR']

def initialize_logic():
    if os.path.exists(MODEL_PATH):
        print("⚡ PHASE 1: Loading Pre-Trained AI Brain... (Instant)")
        return joblib.load(MODEL_PATH)
    
    print("🚀 PHASE 1: First-time setup. Training on dataset...")
    try:
        df = pd.read_csv('train.csv')
        X = df[FEATURES]
        y = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

        model = RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X, y)

        joblib.dump(model, MODEL_PATH)
        print("✅ Training Complete. Model saved.")
        return model
    except Exception as e:
        print(f"❌ Critical Error in Logic: {e}")
        sys.exit()

brain = initialize_logic()

# ============================================================
# PHASE 2: DESIGN CODE (WITH LOGO HEADER)
# ============================================================
html_ui = """
<!DOCTYPE html>
<html>
<head>
<style>
:root {
    --primary: #3b82f6;
    --accent: #ef4444;
    --bg: #0f172a;
    --card: #1e293b;
}

body {
    background: var(--bg);
    color: #f8fafc;
    font-family: 'Inter', system-ui, sans-serif;
    margin: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

/* Header with logo */
.header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 15px;
    justify-content: center;
}

.header img {
    width: 38px;
    height: 38px;
}

.header-title {
    font-size: 24px;
    font-weight: 700;
    color: #60a5fa;
}

.app-window {
    width: 440px;
    background: var(--card);
    border-radius: 28px;
    padding: 40px;
    box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
}

.tagline {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 25px;
    text-transform: uppercase;
}

.input-row { display: flex; gap: 12px; margin-bottom: 12px; }
.input-group { flex: 1; margin-bottom: 15px; }

label {
    display: block;
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 5px;
}

input {
    width: 100%;
    padding: 12px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    color: white;
    outline: none;
}

input:focus {
    border-color: var(--primary);
}

.btn-analyze {
    width: 100%;
    padding: 16px;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 700;
    cursor: pointer;
    margin-top: 20px;
}

.result-panel {
    margin-top: 30px;
    padding: 20px;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.5);
    text-align: center;
    display: none;
}

#res-pct { font-size: 44px; font-weight: 800; }
#res-status { font-size: 13px; font-weight: 700; }
</style>
</head>

<body>
<div class="app-window">

    <div class="header">
        <img src="logo.png">
        <div class="header-title">HeartGuard AI</div>
    </div>

    <p class="tagline">Advanced Neural Analytics</p>

    <div class="input-group">
        <label>Patient Age</label>
        <input type="number" id="Age" value="45">
    </div>

    <div class="input-row">
        <div class="input-group">
            <label>Sex (1:M, 0:F)</label>
            <input type="number" id="Sex" value="1">
        </div>
        <div class="input-group">
            <label>Pain Type (1-4)</label>
            <input type="number" id="CP" value="2">
        </div>
    </div>

    <div class="input-row">
        <div class="input-group">
            <label>Blood Pressure</label>
            <input type="number" id="BP" value="130">
        </div>
        <div class="input-group">
            <label>Cholesterol</label>
            <input type="number" id="Chol" value="240">
        </div>
    </div>

    <div class="input-group">
        <label>Max Heart Rate</label>
        <input type="number" id="HR" value="155">
    </div>

    <button class="btn-analyze" onclick="requestAI()">RUN AI DIAGNOSTIC</button>

    <div id="result-panel" class="result-panel">
        <div id="res-status">ANALYSIS RESULT</div>
        <div id="res-pct">0%</div>
    </div>
</div>

<script>
async function requestAI() {
    const data = {
        'Age': parseFloat(document.getElementById('Age').value),
        'Sex': parseFloat(document.getElementById('Sex').value),
        'Chest pain type': parseFloat(document.getElementById('CP').value),
        'Cholesterol': parseFloat(document.getElementById('Chol').value),
        'BP': parseFloat(document.getElementById('BP').value),
        'Max HR': parseFloat(document.getElementById('HR').value)
    };

    const result = await pywebview.api.predict(data);

    const panel = document.getElementById('result-panel');
    const pct = document.getElementById('res-pct');
    const status = document.getElementById('res-status');

    panel.style.display = 'block';
    pct.innerText = result + '%';

    if(result > 70){
        status.innerText = 'CRITICAL RISK';
        pct.style.color = '#ef4444';
    } else if(result > 35){
        status.innerText = 'MODERATE RISK';
        pct.style.color = '#fbbf24';
    } else {
        status.innerText = 'STABLE CONDITION';
        pct.style.color = '#22c55e';
    }
}
</script>
</body>
</html>
"""

# ============================================================
# PHASE 3: CONNECTIVITY
# ============================================================
class HeartAPI:
    def predict(self, inputs):
        df_input = pd.DataFrame([inputs], columns=FEATURES)
        risk = brain.predict_proba(df_input)[0][1]
        return round(risk * 100, 2)

# ============================================================
# PHASE 4: DEPLOYMENT
# ============================================================
if __name__ == '__main__':
    api = HeartAPI()

    window = webview.create_window(
    "HeartGuard AI Professional",   # 👈 App name at top-left
    html=html_ui,
    js_api=api,
    width=1200,
    height=750,
    resizable=True,      # Enables maximize button
    fullscreen=False,    # MUST be False
    frameless=False      # Shows title bar with Min/Max/Close
)


    webview.start()
